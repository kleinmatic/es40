/* ES40 emulator -- JIT engine implementation. */
#ifdef ES40_JIT

#include "jitengine.h"
#include <cstdio>
#include <cstring>
#define ASMJIT_STATIC
#include <asmjit/x86.h>

namespace {

enum SafeOp {
  OP_NONE,
  OP_ADDQ, OP_SUBQ, OP_ADDL, OP_SUBL,
  OP_S4ADDQ, OP_S8ADDQ, OP_S4SUBQ, OP_S8SUBQ,
  OP_S4ADDL, OP_S8ADDL, OP_S4SUBL, OP_S8SUBL,   // INTA (0x10) scaled longword: sext32((Ra*scale) +/- Rb)
  OP_CMPBGE,                                    // INTA (0x10) per-byte unsigned compare -> 8-bit mask
  OP_SEXTB, OP_SEXTW,                            // FPTI (0x1c) CIX/BWX: Rc = sign-extend byte/word of op2
  OP_ITOFS, OP_ITOFF, OP_ITOFT,                  // ITFP (0x14) int->FP reg moves: f[Fc] = fmt(Ra)
  OP_FTOIS, OP_FTOIT,                            // FPTI (0x1c) FP->int reg moves: Rc = fmt(f[Fa])
  OP_FLTL,                                       // FLTL (0x17) non-arithmetic: FPCR moves, CPYSx, FCMOVx, CVTLQ/QL
  OP_AND, OP_BIS, OP_XOR, OP_BIC, OP_ORNOT, OP_EQV,
  OP_CMOV,                       // INTL (0x11) conditional moves CMOVxx: Rc = cond(Ra) ? op2 : Rc
  OP_AMASK, OP_IMPLVER,          // INTL (0x11) CPU feature probes: Rc = op2 & ~CPU_AMASK / Rc = CPU_IMPLVER
  OP_CMPEQ, OP_CMPLT, OP_CMPLE, OP_CMPULT, OP_CMPULE,
  OP_SLL, OP_SRL, OP_SRA, OP_MULQ,
  OP_MULL, OP_UMULH,             // INTM (0x13): MULL = sext32(Ra*op2); UMULH = hi64 of unsigned Ra*op2
  OP_EXTL, OP_EXTH, OP_INSL, OP_INSH, OP_MSKL, OP_MSKH, OP_ZAP,   // INTS (0x12) byte-manip (Rb&7 keyed)
  OP_NOP, OP_MFENCE,             // MISC (0x18): prefetch/cache hints (no-op), barriers (mfence)
  OP_RPCC, OP_RC, OP_RS,         // MISC (0x18) state reads (Ra dest): cycle counter / read-and-clear,set intr flag
  OP_LDQ, OP_LDL,                // memory-format loads: Ra = MEM[Rb + disp16]
  OP_LDBU, OP_LDWU,              // BWX byte/word loads (0x0a/0x0c): Ra = zero-extend MEM[Rb+disp16].{b,w}
  OP_LDL_L, OP_LDQ_L,            // load-locked (0x2a/0x2b): Ra = MEM[Rb+disp16] + establish LL/SC monitor
  OP_STL_C, OP_STQ_C,            // store-conditional (0x2e/0x2f): cond store Ra -> MEM; Ra = success(1)/fail(0)
  OP_STQ, OP_STL,                // memory-format stores: MEM[Rb + disp16] = Ra
  OP_STB, OP_STW,                // BWX byte/word stores (0x0e/0x0d): MEM[Rb + disp16].{b,w} = Ra
  OP_LDQ_U, OP_STQ_U,            // unaligned quad (0x0b/0x0f): MEM[(Rb+disp16) & ~7] load/store
  OP_LDA, OP_LDAH,               // load-address: Ra = Rb + disp16 (<<16 for LDAH); pure ALU
  OP_HW_MFPR,                    // HW_MFPR (0x19), PALmode only: Ra = IPR[(ins>>8)&0xff] via helper
  OP_HW_LDL, OP_HW_LDQ,          // HW_LD (0x1b) physical func 0/1, PALmode only: Ra = phys[Rb+disp12]
  OP_HW_LDQ_VPTE,                // HW_LD (0x1b) func 5: virtual PTE fetch, access-checked vs KERNEL mode
  OP_HW_MTPR,                    // HW_MTPR (0x1d) side-effect-free IPRs, PALmode only: IPR[fn] = Rb
  OP_HW_STL, OP_HW_STQ,          // HW_ST (0x1f) physical func 0/1, PALmode only: phys[Rb+disp12] = Ra
  OP_JMP,                        // JMP/JSR/RET (0x1a): Ra = PC+4; PC = Rb & ~3 (computed target)
  OP_HW_RET,                     // HW_RET (0x1e), PALmode: PC = Rb & ~2 (computed jump, the PAL return)
  OP_CALL_PAL,                   // CALL_PAL (0x00): save R23/exc_addr; PC = pal_base | entry offset
  // Branch-format terminators (contiguous; see is_branch). Conditional on Ra, plus BR/BSR.
  OP_BEQ, OP_BNE, OP_BLT, OP_BLE, OP_BGT, OP_BGE, OP_BLBC, OP_BLBS, OP_BR, OP_BSR
};

static inline bool is_branch(SafeOp op) { return op >= OP_BEQ && op <= OP_BSR; }
static inline bool is_store(SafeOp op)  { return op == OP_STQ || op == OP_STL; }
// A terminator ends the block and writes its own next PC (branches + the computed jump).
static inline bool is_terminator(SafeOp op) { return op == OP_JMP || op == OP_HW_RET || op == OP_CALL_PAL || is_branch(op); }

// Safe = goto-free, register-only operate-format ops (no trap, memory, or branch).
// pal_block enables PALmode-only ops (HW_MFPR): outside PALmode they'd OPCDEC, so only
// compile them when the block is PALmode (the dispatcher keys blocks by PC bit 0).
SafeOp classify(uint32_t ins, bool pal_block)
{
  uint32_t opcode = ins >> 26;
  uint32_t func = (ins >> 5) & 0x7F;
  switch (opcode) {
    case 0x10: // INTA
      switch (func) {
        case 0x20: return OP_ADDQ;   case 0x29: return OP_SUBQ;
        case 0x00: return OP_ADDL;   case 0x09: return OP_SUBL;
        case 0x22: return OP_S4ADDQ; case 0x32: return OP_S8ADDQ;
        case 0x2b: return OP_S4SUBQ; case 0x3b: return OP_S8SUBQ;
        case 0x02: return OP_S4ADDL; case 0x12: return OP_S8ADDL;
        case 0x0b: return OP_S4SUBL; case 0x1b: return OP_S8SUBL;
        case 0x0f: return OP_CMPBGE;
        case 0x2d: return OP_CMPEQ;  case 0x4d: return OP_CMPLT;
        case 0x6d: return OP_CMPLE;  case 0x1d: return OP_CMPULT;
        case 0x3d: return OP_CMPULE;
        // ADDL/V (0x40), SUBL/V (0x49), ADDQ/V (0x60), SUBQ/V (0x69): overflow-trapping -> interpret
      }
      break;
    case 0x11: // INTL
      switch (func) {
        case 0x00: return OP_AND;   case 0x20: return OP_BIS;
        case 0x40: return OP_XOR;   case 0x08: return OP_BIC;
        case 0x28: return OP_ORNOT; case 0x48: return OP_EQV;
        case 0x14: case 0x16: case 0x24: case 0x26:   // CMOVLBS/LBC/EQ/NE
        case 0x44: case 0x46: case 0x64: case 0x66:   // CMOVLT/GE/LE/GT
          return OP_CMOV;
        case 0x61:                                    // AMASK: Ra must be R31 (else the interp's
          // DO_AMASK traps OPCDEC) -- compile only the architecturally valid form.
          if (((ins >> 21) & 0x1f) != 31) return OP_NONE;
          return OP_AMASK;
        case 0x6c: return OP_IMPLVER;                 // IMPLVER: Rc = implementation version constant
      }
      break;
    case 0x12: // INTS: shifts + byte-manipulation (extract / insert / mask / zap), Rb&7 keyed
      switch (func) {
        case 0x39: return OP_SLL; case 0x34: return OP_SRL; case 0x3c: return OP_SRA;
        case 0x06: case 0x16: case 0x26: case 0x36: return OP_EXTL;   // EXTBL/WL/LL/QL
        case 0x5a: case 0x6a: case 0x7a:            return OP_EXTH;   // EXTWH/LH/QH
        case 0x0b: case 0x1b: case 0x2b: case 0x3b: return OP_INSL;   // INSBL/WL/LL/QL
        case 0x57: case 0x67: case 0x77:            return OP_INSH;   // INSWH/LH/QH
        case 0x02: case 0x12: case 0x22: case 0x32: return OP_MSKL;   // MSKBL/WL/LL/QL
        case 0x52: case 0x62: case 0x72:            return OP_MSKH;   // MSKWH/LH/QH
        case 0x30: case 0x31:                       return OP_ZAP;    // ZAP / ZAPNOT
      }
      break;
    case 0x13: // INTM (integer multiply). MULL/V (0x40) + MULQ/V (0x60) overflow-trap -> interpret.
      if (func == 0x20) return OP_MULQ;
      if (func == 0x00) return OP_MULL;    // 32-bit multiply, low 32 sign-extended
      if (func == 0x30) return OP_UMULH;   // high 64 bits of the unsigned 128-bit product
      break;
    case 0x14: {  // ITFP: compile the int->FP register moves; SQRT* (FP math: FPCR/rounding/traps)
      // stays interpreted. Rb must be 31 (else the interp traps OPCDEC); Fc==31 is interpreted
      // (the interp zeroes f[31] per instruction -- a compiled write would desync).
      const uint32_t f14 = (ins >> 5) & 0x7ff;
      if (((ins >> 16) & 0x1f) != 31 || (ins & 0x1f) == 31) return OP_NONE;
      if (f14 == 0x004) return OP_ITOFS;
      if (f14 == 0x014) return OP_ITOFF;
      if (f14 == 0x024) return OP_ITOFT;
      return OP_NONE;
    }
    case 0x17: {  // FLTL: all non-arithmetic (FPCR moves, sign-copies, FCMOVs, CVTLQ/QL) via
      // jit_fltl; only the /V trap variants of CVTQL stay interpreted.
      const uint32_t f17 = (ins >> 5) & 0x7ff;
      const bool ok = f17 == 0x010 || (f17 >= 0x020 && f17 <= 0x022) || f17 == 0x024
                   || f17 == 0x025 || (f17 >= 0x02a && f17 <= 0x02f) || f17 == 0x030;
      if (!ok) return OP_NONE;
      // f31-dest gate (interp zeroes f[31] per instr): MF_FPCR writes f[Fa]; MT_FPCR has no FP dest
      if (f17 == 0x025)      { if (((ins >> 21) & 0x1f) == 31) return OP_NONE; }
      else if (f17 != 0x024) { if ((ins & 0x1f) == 31) return OP_NONE; }
      return OP_FLTL;
    }
    case 0x1c: // FPTI (CIX/BWX/MVI/FP-moves): the pure-ALU sign-extends and the FP->int register
      // moves compile here. CTPOP/CTLZ/CTTZ (bit-count: zero-handling + popcnt/lzcnt CPU feature)
      // and MVI packed media (PERR/MIN/MAX/PK/UNPK) stay interpreted for now.
      if (func == 0x00) return OP_SEXTB;   // sign-extend byte of op2
      if (func == 0x01) return OP_SEXTW;   // sign-extend word of op2
      if (func == 0x70 || func == 0x78) {  // FTOIT / FTOIS: Rb must be 31; Rc==31 -> interpret
        if (((ins >> 16) & 0x1f) != 31 || (ins & 0x1f) == 31) return OP_NONE;
        return (func == 0x70) ? OP_FTOIT : OP_FTOIS;
      }
      break;
    case 0x18: // MISC: memory barriers -> mfence (keep MP ordering); prefetch/cache hints -> no-op
      switch (ins & 0xFFFF) {
        case 0x0000: case 0x0400:                    // TRAPB, EXCB
        case 0x4000: case 0x4400: return OP_MFENCE;  // MB, WMB
        case 0x8000: case 0xA000: case 0xE800:       // FETCH, FETCH_M, ECB
        case 0xF800: case 0xFC00: return OP_NOP;     // WH64, WH64EN
        // RPCC/RC/RS read time-varying / consumed state into Ra -- compile the value-returning
        // forms (Ra!=31) so the verify can log+replay the read; Ra==31 (no GPR dest; RC/RS still
        // side-effect the flag) falls through to the interpreter.
        case 0xC000: if (((ins >> 21) & 0x1f) != 31) return OP_RPCC; break;  // RPCC (cycle counter)
        case 0xE000: if (((ins >> 21) & 0x1f) != 31) return OP_RC;   break;  // RC (read & clear flag)
        case 0xF000: if (((ins >> 21) & 0x1f) != 31) return OP_RS;   break;  // RS (read & set flag)
      }
      break;
    case 0x00: {                // CALL_PAL: compile valid standard funcs (priv 0x00-0x3f, unpriv 0x80-0xbf)
      const uint32_t fn = ins & 0x1FFFFFFF;
      if (fn <= 0x3F || (fn >= 0x80 && fn <= 0xBF)) return OP_CALL_PAL;
      break;                    // SRM specials (0x1234xx) / invalid ranges -> interpret
    }
    case 0x08: return OP_LDA;   // load address (Ra = Rb + disp16) -- pure ALU, no memory
    case 0x09: return OP_LDAH;  // load address high (Ra = Rb + (disp16 << 16))
    case 0x19: {                // HW_MFPR: read IPR -> Ra. PALmode-only (else OPCDEC).
      // ISUM (fn 0x0d) reads async interrupt lines -- compiled via log/replay. Only the IPRs
      // jit_hw_mfpr implements compile: an unknown function must reach the interpreter's
      // UNKNOWN2 (OPCDEC trap), which the helper's keep-Ra default would silently skip.
      if (!pal_block) return OP_NONE;
      const uint32_t fn = (ins >> 8) & 0xff;
      const bool known = ((fn & 0xc0) == 0x40)                                   // PCTX group
                      || (fn >= 0x05 && fn <= 0x0d) || fn == 0x0f || fn == 0x10
                      || fn == 0x11 || fn == 0x14 || fn == 0x16 || fn == 0x27
                      || fn == 0x2a || fn == 0x2b || fn == 0xc0 || fn == 0xc2 || fn == 0xc3;
      return known ? OP_HW_MFPR : OP_NONE;
    }
    case 0x1b: {                // HW_LD: read phys[Rb+disp12] -> Ra. PALmode-only. Compile the
      // physical forms (func 0/1) and the quad VPTE fetch (func 5, kernel-checked -- see
      // jit_read_vpte). Locked, virtual, alt and write-check forms stay interpreted.
      if (!pal_block) return OP_NONE;
      const uint32_t f = (ins >> 12) & 0xf;
      if (f == 0) return OP_HW_LDL;
      if (f == 1) return OP_HW_LDQ;
      if (f == 5) {             // Ra==31: nothing to log/replay -> interpret
        if (((ins >> 21) & 0x1f) == 31) return OP_NONE;
        return OP_HW_LDQ_VPTE;
      }
      return OP_NONE;
    }
    case 0x1d: {                // HW_MTPR (PALmode): compile the pure-store IPRs, the TB fills
      // (idempotent forwards to add_tb_i/_d) and IER (field stores + check_int kick). TB/cache
      // flushes, CM writes, SIRR/HW_INT_CLR, PAL_BASE, i_ctl, the 0x40-7f group stay interpreted.
      if (!pal_block) return OP_NONE;
      switch ((ins >> 8) & 0xff) {
        case 0x00: case 0x14: case 0x20: case 0x26:   // ITB_TAG, PCTR_CTL, DTB_TAG0, DTB_ALTMODE
        case 0x29: case 0xa0: case 0xc0:              // DC_CTL, DTB_TAG1, CC
        case 0x01: case 0x21: case 0xa1:              // ITB_PTE, DTB_PTE0, DTB_PTE1 (TB fills)
        case 0x0a:                                    // IER (interrupt enables + check_int kick)
          return OP_HW_MTPR;
        case 0x15: case 0x17: case 0x27:              // CLR_MAP, SLEEP, MM_STAT (no-ops)
        case 0x2b: case 0x2c: case 0x2d:              // C_DATA, C_SHIFT, M_FIX (no-ops)
          return OP_NOP;
      }
      return OP_NONE;
    }
    case 0x1f: {                // HW_ST (PALmode): compile physical longword/quadword (func 0/1).
      // Store-conditional (2/3) does LL/SC, virtual (4/5) and virtual-alt (12/13) translate with
      // side effects -> interpret. Mirrors HW_LD; the value is Ra, base Rb, displacement 12-bit.
      if (!pal_block) return OP_NONE;
      const uint32_t f = (ins >> 12) & 0xf;
      if (f == 0) return OP_HW_STL;
      if (f == 1) return OP_HW_STQ;
      return OP_NONE;
    }
    case 0x1a: return OP_JMP;   // JMP/JSR/RET: computed jump (target = Rb & ~3). Now compiled --
                                // chained in-frame via a block-cache lookup (jit_indirect), which
                                // handles varying targets without the old single-slot thrash.
    case 0x1e:                  // HW_RET (HWREI): a PAL return, also a simple computed jump (Rb & ~2).
      return pal_block ? OP_HW_RET : OP_NONE;
    case 0x28: return OP_LDL;   // memory-format loads (Ra = MEM[Rb+disp16])
    case 0x29: return OP_LDQ;
    case 0x0a: return OP_LDBU;  // BWX byte/word loads (Ra = zero-extend MEM[Rb+disp16].{b,w})
    case 0x0c: return OP_LDWU;
    case 0x0b: return OP_LDQ_U;  // unaligned quad load: Ra = MEM[(Rb+disp16) & ~7]
    case 0x2a:                  // LDL_L / LDQ_L: the load-locked half of LL/SC. Compile the value-
    case 0x2b:                  // returning forms only -- Ra==31 (lock without consuming the value)
      // leaves the loaded value out of the GPRs, so the verify can't replay it; interpret those.
      if (((ins >> 21) & 0x1f) == 31) return OP_NONE;
      return (opcode == 0x2a) ? OP_LDL_L : OP_LDQ_L;
    case 0x2c: return OP_STL;   // memory-format stores (MEM[Rb+disp16] = Ra)
    case 0x2d: return OP_STQ;
    case 0x0e: return OP_STB;   // BWX byte/word stores (MEM[Rb+disp16].{b,w} = Ra low bits)
    case 0x0d: return OP_STW;
    case 0x0f: return OP_STQ_U;  // unaligned quad store: MEM[(Rb+disp16) & ~7] = Ra
    case 0x2e:                  // STL_C / STQ_C: the store-conditional half of LL/SC. Ra is both the
    case 0x2f:                  // value AND the success/fail dest. Compile Ra!=31 (Ra==31 discards the
      // result into R31, so the verify can't capture it from a GPR); interpret those.
      if (((ins >> 21) & 0x1f) == 31) return OP_NONE;
      return (opcode == 0x2e) ? OP_STL_C : OP_STQ_C;
    // Branch format: opcode | Ra | disp21. Conditional on Ra, plus BR/BSR.
    case 0x30: return OP_BR;    case 0x34: return OP_BSR;
    case 0x38: return OP_BLBC;  case 0x39: return OP_BEQ;
    case 0x3a: return OP_BLT;   case 0x3b: return OP_BLE;
    case 0x3c: return OP_BLBS;  case 0x3d: return OP_BNE;
    case 0x3e: return OP_BGE;   case 0x3f: return OP_BGT;
  }
  return OP_NONE;
}

} // namespace

// Defined further down; forward-declared so compile_block's punch-list print can use it.
static const char* opcode_name(unsigned op);

CJitEngine::CJitEngine(int cpu_id) : m_cpu_id(cpu_id), m_recorded(0), m_code_bytes(0), m_rt(nullptr)
{
  memset(m_blocks, 0, sizeof(m_blocks));    // flush() is lazy (gen bump) -- zero the slots here
  m_rt = new asmjit::JitRuntime();
#ifdef JIT_VERIFY
  m_v_exec = m_v_fail = 0;
#endif
#ifdef JIT_STATS
  m_stat_native = m_stat_interp = m_stat_hot = m_stat_miss = 0;
  m_stat_compiled = m_stat_plen_sum = 0;
  memset(m_term_op, 0, sizeof(m_term_op));
  memset(m_pal_func, 0, sizeof(m_pal_func));
  memset(m_mtpr_func, 0, sizeof(m_mtpr_func));
  memset(m_hwld_func, 0, sizeof(m_hwld_func));
  m_first_breaker_logged = false;
#endif
}

CJitEngine::~CJitEngine()
{
  delete (asmjit::JitRuntime*) m_rt;
}

// FNV-1a/64 over a block's source words -- record() uses it to revalidate kept code after a flush
// instead of recompiling, and to catch self-modifying code (changed bytes -> different hash).
static inline uint64_t src_hash(const uint8_t* p, uint32_t n_instr)
{
  const uint32_t* w = (const uint32_t*) p;
  uint64_t h = 1469598103934665603ULL;
  for (uint32_t i = 0; i < n_instr; ++i) { h ^= w[i]; h *= 1099511628211ULL; }
  return h;
}

CJitEngine::JitBlock* CJitEngine::record(uint64_t virt_pc, uint64_t phys_pc, uint32_t asn, bool asm_global, uint32_t n_instr, const uint8_t* dram)
{
  JitBlock& b = m_blocks[index_of(virt_pc)];
  // record() is only reached after the dispatcher validated the live physical, and every returning
  // branch below verifies the code bytes (flush-fresh or hash), so stamp the chain epoch here.
  b.vgen = m_itb_gen + m_flush_gen;
  // Still valid + matching + flush-fresh: nothing flushed us since last seen, so the code can't
  // have changed. (flush_gen-stale must NOT take this no-hash path -- post-IMB needs the hash.)
  if (b.valid && b.flush_gen == m_flush_gen && b.tag == virt_pc && (b.asm_global || b.asn == asn)
      && b.phys == phys_pc) {
    b.n_instr = n_instr;
    return &b;
  }
  // Revalidate: a flush dropped the block but kept the compiled code. If the bytes the prefix was
  // compiled from still hash the same, reuse it instead of recompiling. Hash over hash_len, NOT
  // the caller's n_instr -- interrupt-truncated cold passes make n vary for the same block.
  if (b.code && b.tag == virt_pc && (b.asm_global || b.asn == asn) && b.phys == phys_pc
      && b.src_sum == src_hash(dram + phys_pc, b.hash_len)) {
    b.valid = true;
    b.flush_gen = m_flush_gen;
    b.n_instr = n_instr;
    b.jit_body = (void*) ((uint8_t*) (void*) b.code + b.body_off);   // restore chained re-entry
    return &b;
  }
  // New block, page remap, or modified bytes: record fresh and force a recompile.
  b.tag = virt_pc;
  b.phys = phys_pc;
  b.asn = asn;
  b.asm_global = asm_global;
  b.n_instr = n_instr;
  b.valid = true;
  b.flush_gen = m_flush_gen;
  b.code = nullptr;
  b.jit_body = nullptr;   // not compiled yet -> cached links to us must miss until compile
  b.link = nullptr;       // no cached successor yet
  b.prefix_len = 0;
  b.compiled = false;
  if (++m_recorded == 50000)
    printf("[JIT][CPU%d] block dispatcher active: 50000 blocks discovered.\n", m_cpu_id);
  return &b;
}

// Lazy-flush survivor: the dispatcher calls this on a lookup miss, with the LIVE physical it just
// translated. If the slot matches and its source bytes still hash the same, restamp and return it
// straight to the hot path 
CJitEngine::JitBlock* CJitEngine::revalidate_flushed(uint64_t virt_pc, uint32_t asn, uint64_t phys_pc, const uint8_t* dram)
{
  JitBlock& b = m_blocks[index_of(virt_pc)];
  if (!(b.valid && b.code && b.tag == virt_pc && (b.asm_global || b.asn == asn)))
    return nullptr;
  if (b.phys != phys_pc || b.src_sum != src_hash(dram + phys_pc, b.hash_len))
    return nullptr;
  b.flush_gen = m_flush_gen;
  b.vgen = m_itb_gen + m_flush_gen;   // phys + code bytes just validated
  b.jit_body = (void*) ((uint8_t*) (void*) b.code + b.body_off);
  return &b;
}

// Free ALL compiled code (delete+new of the runtime; reset()-and-reuse corrupted the JitAllocator
// block tree) and drop every slot's now-dangling pointers. Safe only from this CPU's cold path
// (never while its compiled code could be executing); runtimes are per-CPU.
void CJitEngine::reclaim_code()
{
  //printf("[JIT][CPU%d] code reclaim: %llu MB freed\n", m_cpu_id,
  //       (unsigned long long) (m_code_bytes >> 20));
  delete (asmjit::JitRuntime*) m_rt;
  m_rt = new asmjit::JitRuntime();
  m_code_bytes = 0;
  for (int i = 0; i < kCacheEntries; ++i) {
    m_blocks[i].valid = false;
    m_blocks[i].code = nullptr;
    m_blocks[i].jit_body = nullptr;
    m_blocks[i].compiled = false;
  }
}

void CJitEngine::flush()
{
  // LAZY:  don't walk 16K slots each time. Bump the generation instead: stale blocks miss in
  // lookup() and revalidate_flushed() re-hashes their source bytes before they run again.
  ++m_flush_gen;
  if (m_rt && m_code_bytes >= kReclaimBytes)
    reclaim_code();
}

// ASM-bit-clear icache flush (process/ASN switch): drop only !asm_global blocks. Global (ASM) PAL
// blocks are ASN-independent and must survive it. Past the reclaim cap, defer to the full flush().
void CJitEngine::flush_non_global()
{
  if (m_rt && m_code_bytes >= kReclaimBytes) { flush(); return; }
  for (int i = 0; i < kCacheEntries; ++i) {
    if (!m_blocks[i].asm_global) {
      m_blocks[i].valid = false;
      m_blocks[i].jit_body = nullptr;
    }
  }
}

// ZAP/ZAPNOT byte-expand: g_zapnot_mask[b] has byte i = 0xFF where bit i of b is set (ZAPNOT keeps
// those bytes; ZAP keeps the complement). Compiled ZAP indexes this instead of an 8-way bit test.
static uint64_t g_zapnot_mask[256];
static const bool g_zapnot_init = [] {
  for (int b = 0; b < 256; ++b) {
    uint64_t m = 0;
    for (int i = 0; i < 8; ++i) if (b & (1 << i)) m |= (uint64_t) 0xff << (i * 8);
    g_zapnot_mask[b] = m;
  }
  return true;
}();

void CJitEngine::compile_block(JitBlock* b, const uint8_t* dram, uint64_t dram_size, void* read_helper, void* write_helper, void* opcdec_helper, void* hw_mfpr_helper, void* hw_ld_helper, void* hw_mtpr_helper, void* hw_st_helper, void* indirect_helper, void* read_locked_helper, void* stc_helper, void* misc_helper, void* read_vpte_helper, void* itof_helper, void* ftoi_helper, void* fltl_helper)
{
  using namespace asmjit;
  // Reclaim must self-trigger here, NOT only in flush(): flush() runs when the guest executes
  // IMB/IC_FLUSH, and a compute-heavy phase can go minutes without one while recompiles keep
  // allocating -- code memory grew unbounded (multi-GB). Safe: we're in this CPU's cold path.
  if (m_rt && m_code_bytes >= kReclaimBytes) {
    reclaim_code();
    b->valid = true;   // b was just (re)validated by record(); restore it after the wipe
  }
  b->compiled = true;

  uint64_t phys = b->phys;
  if (b->n_instr == 0 || phys + (uint64_t) b->n_instr * 4 > dram_size) return;
  const uint32_t* words = (const uint32_t*) (dram + phys);   // x86 LE == Alpha LE

  // PALmode blocks (PC bit 0) remap R4-7/R20-23 to the shadow bank (see RREG); reg() applies it.
  const bool pal_block = (b->tag & 1) != 0;

  // Stop at the 8 KB page boundary: past it the next instruction's physical
  // address need not be phys+4 (the next virtual page maps elsewhere), so words[]
  // there would be the wrong instructions. (The page-crossing case verify caught.)
  const uint64_t page_end = (phys & ~(uint64_t) 0x1FFF) + 0x2000;
  uint32_t plen = 0;
  bool terminator_branch = false;       // last instruction is a compiled terminator (sets its own PC)
  bool terminator_jmp = false;          // ...and it's a computed jump (don't chain: targets vary)
  while (plen < b->n_instr && plen < 64
         && (phys + (uint64_t) plen * 4) < page_end)
  {
    SafeOp sop = classify(words[plen], pal_block);
    if (sop == OP_NONE) {               // uncompilable op ends the straight-line prefix
#ifdef JIT_STATS
      const uint32_t bop = words[plen] >> 26;
      m_term_op[bop]++;                 // tally what cut this block (the coverage gap to chase)
      if (bop == 0x00)                  // CALL_PAL: also tally the function code (low 8 bits)
        m_pal_func[words[plen] & 0xFF]++;
      else if (bop == 0x1d)             // HW_MTPR: tally the IPR index -- which writes break blocks
        m_mtpr_func[(words[plen] >> 8) & 0xFF]++;
      else if (bop == 0x1b)             // HW_LD: tally the form (phys/virt/lock/vpte/chk, ins[15:12])
        m_hwld_func[(words[plen] >> 12) & 0xF]++;
      // Punch list: one-shot print of the first ACTIONABLE breaker -- skip the opcodes whose
      // compilable subset is already settled, so it points at the next NEW target rather than a
      // decided one: 0x00 CALL_PAL (terminator), 0x1b HW_LD / 0x1f HW_ST (physical done;
      // conditional/virtual forms side-effecting), 0x1d HW_MTPR (pure-store IPRs done; rest
      // side-effecting), 0x10 INTA + 0x13 INTM (non-trapping ops done; only /V overflow-trap variants left),
      // 0x18 MISC (barriers/hints + RPCC/RC/RS via log/replay done; only the rare Ra==31 RC/RS forms interpret),
      // 0x14 ITFP (ITOF* moves done; SQRT* = the deferred FP-math class: FPCR/rounding/traps),
      // 0x17 FLTL (non-arithmetic done; only CVTQL/V trap variants left).
      // JMP (0x1a) + HW_RET (0x1e) are now compiled+chained. Stats count all.
      if (!m_first_breaker_logged && bop != 0x00 && bop != 0x1b && bop != 0x1d && bop != 0x1f && bop != 0x10 && bop != 0x18 && bop != 0x13 && bop != 0x14 && bop != 0x17) {
        m_first_breaker_logged = true;
        printf("[JIT][PUNCH][CPU%d] first unhandled breaker: %s(0x%02x) ins=%08x at pc=%016llx%s\n",
               m_cpu_id, opcode_name(bop), bop, words[plen],
               (unsigned long long) ((b->tag & ~(uint64_t) 1) + (uint64_t) plen * 4),
               pal_block ? "  [PALmode]" : "");
      }
#endif
      break;
    }
    plen++;
    if (is_terminator(sop)) {           // branch or computed jump ends the block
      terminator_branch = true;
      if (sop == OP_JMP || sop == OP_HW_RET) terminator_jmp = true;
      break;
    }
  }

  if (plen == 0) return;

  // Emit  uint32_t fn(CAlphaCPU* cpu, uint64_t* regs)  (Win64: cpu=RCX, regs=RDX).
  // Keep cpu in RSI and regs in RBX (callee-saved, so they survive helper calls);
  // reserve a 40-byte frame (32 shadow + 8 load-out slot) that keeps RSP 16-aligned
  // for calls. RAX = op1/result, RCX = operand2 (CL for variable shifts).
  CodeHolder code;
  if (code.init(((JitRuntime*) m_rt)->environment()) != Error::kOk) return;
  x86::Assembler a(&code);
  a.push(x86::rbx);
  a.push(x86::rsi);
  a.push(x86::r14);            // callee-saved: accumulates the chain's instruction count
  a.sub(x86::rsp, imm(48));    // 32 shadow + load-out slot; keeps RSP 16-aligned at calls
  a.mov(x86::rsi, x86::rcx);   // cpu
  a.mov(x86::rbx, x86::rdx);   // regs base
  a.xor_(x86::r14d, x86::r14d);   // instruction count := 0

  Label done = a.new_label();  // shared exit: restore frame + ret (EAX preset by caller)
  Label body = a.new_label();  // chained re-entry (after the prologue; preserves R14)
  a.bind(body);
  const size_t body_off = code.code_size();   // byte offset of the chained entry from fn

  // The compiled block computes its own next PC into state.pc at every exit (the
  // foundation for branch compilation and block linking). R10 is scratch here.
  auto set_pc = [&](uint64_t pc_val) {
    a.mov(x86::r10, imm(pc_val));
    a.mov(x86::qword_ptr(x86::rsi, m_off.state_pc), x86::r10);
  };

  for (uint32_t i = 0; i < plen; ++i) {
    uint32_t ins = words[i];
    SafeOp op = classify(ins, pal_block);
    int ra = (ins >> 21) & 0x1F;
    int rb = (ins >> 16) & 0x1F;
    int rc = ins & 0x1F;
    bool islit = ((ins >> 12) & 1) != 0;
    uint32_t lit = (ins >> 13) & 0xFF;

    // MISC (0x18) barriers/hints: emit an mfence for TRAPB/EXCB/MB/WMB (x86's seq_cst fence, to
    // preserve the guest's MP memory ordering, matching DO_*'s atomic_thread_fence), nothing for
    // the prefetch/cache hints -- then keep going so the block extends straight past them.
    if (op == OP_NOP)    continue;
    if (op == OP_MFENCE) { a.mfence(); continue; }

    auto reg = [&](int r) {
      // PALshadow (RREG, AlphaCPU.h): in a PALmode block with SDE set, R4-7 and R20-23 map to
      // the shadow bank r[r+32]. 
      int idx = (pal_block && ((r & 0xc) == 0x4)) ? r + 32 : r;
      return x86::qword_ptr(x86::rbx, idx * 8);
    };
    // 32-bit (low-dword) view with the SAME shadow remap, for the 32-bit ALU ops (ADDL/SUBL).
    auto reg32 = [&](int r) {
      int idx = (pal_block && ((r & 0xc) == 0x4)) ? r + 32 : r;
      return x86::dword_ptr(x86::rbx, idx * 8);
    };
    auto op1_rax = [&]() {
      if (ra == 31) a.xor_(x86::eax, x86::eax);
      else          a.mov(x86::rax, reg(ra));
    };
    auto op2_rcx = [&]() {                  // operand2 (literal, or r[Rb] with r31=0)
      if (islit)         a.mov(x86::rcx, imm(lit));
      else if (rb == 31) a.xor_(x86::ecx, x86::ecx);
      else               a.mov(x86::rcx, reg(rb));
    };

    // Memory-format loads: va = regs[Rb] + disp16.
    if (op == OP_LDQ || op == OP_LDL || op == OP_LDBU || op == OP_LDWU || op == OP_LDQ_U) {
      if (ra == 31) continue;            // LDx R31 is a NOP (interpreter skips the read)
      const int disp = (int) (int16_t) (ins & 0xFFFF);
      const int size_bits = (op == OP_LDQ || op == OP_LDQ_U) ? 64 : (op == OP_LDL) ? 32 : (op == OP_LDWU) ? 16 : 8;
      const int amask = (size_bits / 8) - 1;
      if (rb == 31)  a.mov(x86::rdx, imm(disp));         // va -> RDX
      else        {  a.mov(x86::rdx, reg(rb)); if (disp) a.add(x86::rdx, imm(disp)); }
      if (op == OP_LDQ_U) a.and_(x86::rdx, imm(~(uint64_t) 7));   // LDQ_U: force 8-byte alignment

      // Slow path: jit_read(cpu, va, size, &out); on fault bail to `done` returning i
      // (0..i-1 committed). In JIT_VERIFY builds this is the ONLY path, so the helper's
      // replay keeps the differential check race-free.
      auto emit_helper = [&]() {
        a.mov(x86::rcx, x86::rsi);                       // cpu
        a.mov(x86::r8d, imm(size_bits));                 // size in bits
        a.lea(x86::r9, x86::qword_ptr(x86::rsp, 32));    // &out slot
        a.mov(x86::rax, imm((uint64_t) read_helper));
        a.call(x86::rax);
        Label ok = a.new_label();
        a.test(x86::eax, x86::eax);
        a.jz(ok);
        set_pc(b->tag + 4 * (uint64_t) i);               // resume at the faulting load
        a.mov(x86::eax, imm(i));                          // this iteration: i instrs done
        a.add(x86::eax, x86::r14d);                       // + earlier chained iterations
        a.jmp(done);
        a.bind(ok);
        if (op == OP_LDQ || op == OP_LDQ_U) a.mov(x86::rax, x86::qword_ptr(x86::rsp, 32));  // LDQ/LDQ_U: full quad
        else if (op == OP_LDL)  a.movsxd(x86::rax, x86::dword_ptr(x86::rsp, 32));
        else if (op == OP_LDWU) a.movzx(x86::eax, x86::word_ptr(x86::rsp, 32));   // BWX: zero-extend
        else                    a.movzx(x86::eax, x86::byte_ptr(x86::rsp, 32));   // LDBU
        a.mov(reg(ra), x86::rax);
      };

#ifdef JIT_VERIFY
      emit_helper();
#else
      // Inline fast path: aligned + data_page_cache[0][dpc_index(va)] hit + DRAM. Falls to the
      // helper on misalign / cache miss / MMIO. Mirrors jit_read's data-cache path. RDX = va
      // (preserved); R11 = slot byte offset = dpc_index(va)*stride; RAX/R10 scratch.
      Label slow  = a.new_label();
      Label ldone = a.new_label();
      a.test(x86::dl, imm(amask));                                      a.jnz(slow);
      a.mov(x86::r11, x86::rdx);
      a.shr(x86::r11, imm(13));
      a.and_(x86::r11, imm(m_off.dpc_mask));                            // r11 = dpc_index(va)
      a.imul(x86::r11, x86::r11, imm(m_off.dpc_stride));                // r11 = slot byte offset
      a.mov(x86::r10, x86::rdx);
      a.and_(x86::r10, imm(-0x2000));                                   // r10 = va & ~0x1FFF
      a.cmp(x86::byte_ptr(x86::rsi, x86::r11, 0, m_off.dpc_valid), imm(0));         a.je(slow);
      a.cmp(x86::qword_ptr(x86::rsi, x86::r11, 0, m_off.dpc_virt_page), x86::r10);  a.jne(slow);
      a.mov(x86::eax, x86::dword_ptr(x86::rsi, m_off.state_cm));
      a.cmp(x86::dword_ptr(x86::rsi, x86::r11, 0, m_off.dpc_cm), x86::eax);         a.jne(slow);
      a.mov(x86::eax, x86::dword_ptr(x86::rsi, m_off.state_asn0));
      a.cmp(x86::dword_ptr(x86::rsi, x86::r11, 0, m_off.dpc_asn), x86::eax);        a.jne(slow);
      a.mov(x86::rax, x86::qword_ptr(x86::rsi, x86::r11, 0, m_off.dpc_phys_base));
      a.mov(x86::r10, x86::rdx);
      a.and_(x86::r10, imm(0x1FFF));
      a.or_(x86::rax, x86::r10);                                        // rax = phys
      a.cmp(x86::rax, x86::qword_ptr(x86::rsi, m_off.dram_size));       a.jae(slow);
      a.mov(x86::r10, x86::qword_ptr(x86::rsi, m_off.dram_ptr));
      if (op == OP_LDQ || op == OP_LDQ_U) a.mov(x86::rax, x86::qword_ptr(x86::r10, x86::rax));  // LDQ/LDQ_U: full quad
      else if (op == OP_LDL)  a.movsxd(x86::rax, x86::dword_ptr(x86::r10, x86::rax));
      else if (op == OP_LDWU) a.movzx(x86::eax, x86::word_ptr(x86::r10, x86::rax));   // BWX: zero-extend
      else                    a.movzx(x86::eax, x86::byte_ptr(x86::r10, x86::rax));   // LDBU
      a.mov(reg(ra), x86::rax);
      a.jmp(ldone);
      a.bind(slow);
      emit_helper();
      a.bind(ldone);
#endif
      continue;
    }

    // Memory-format stores: MEM[regs[Rb] + disp16] = regs[Ra]. Via jit_write(cpu,va,size,
    // value): in verify it compares against the interpreter's recorded store (stores touch
    // memory, not GPRs), in production it writes (side-effect-free cache translation, bail
    // on miss). On a fault, bail like a load. (Inline write fast path is a follow-up.)
    if (op == OP_STL || op == OP_STQ || op == OP_STB || op == OP_STW || op == OP_STQ_U) {
      const int disp = (int) (int16_t) (ins & 0xFFFF);
      const int size_bits = (op == OP_STQ || op == OP_STQ_U) ? 64 : (op == OP_STL) ? 32 : (op == OP_STW) ? 16 : 8;
      if (rb == 31)  a.mov(x86::rdx, imm(disp));                       // va -> RDX
      else        {  a.mov(x86::rdx, reg(rb)); if (disp) a.add(x86::rdx, imm(disp)); }
      if (op == OP_STQ_U) a.and_(x86::rdx, imm(~(uint64_t) 7));        // STQ_U: force 8-byte alignment
      if (ra == 31)  a.xor_(x86::r9d, x86::r9d);                       // value -> R9 (R31 == 0)
      else           a.mov(x86::r9, reg(ra));
      a.mov(x86::rcx, x86::rsi);                                       // cpu
      a.mov(x86::r8d, imm(size_bits));                                 // size in bits
      a.mov(x86::rax, imm((uint64_t) write_helper));
      a.call(x86::rax);                                               // jit_write(cpu, va, size, value)
      Label ok = a.new_label();
      a.test(x86::eax, x86::eax);
      a.jz(ok);
      set_pc(b->tag + 4 * (uint64_t) i);                              // resume at the faulting store
      a.mov(x86::eax, imm(i));                                         // this iteration: i instrs done
      a.add(x86::eax, x86::r14d);                                      // + earlier chained iterations
      a.jmp(done);
      a.bind(ok);
      continue;
    }

    // Store-conditional STL_C/STQ_C (0x2e/0x2f): conditionally store Ra; Ra = 1 (success) / 0 (fail).
    // jit_stc consumes the LL monitor + does the host CAS (production), or compares against the
    // interpreter's logged outcome (verify). 
    if (op == OP_STL_C || op == OP_STQ_C) {
      const int disp = (int) (int16_t) (ins & 0xFFFF);
      const int size_bits = (op == OP_STQ_C) ? 64 : 32;
      if (rb == 31)  a.mov(x86::rdx, imm(disp));                       // va -> RDX
      else        {  a.mov(x86::rdx, reg(rb)); if (disp) a.add(x86::rdx, imm(disp)); }
      a.mov(x86::r9, reg(ra));                                         // value (Ra) -> R9
      a.mov(x86::rcx, x86::rsi);                                       // cpu
      a.mov(x86::r8d, imm(size_bits));                                 // size in bits
      a.mov(x86::rax, imm((uint64_t) stc_helper));
      a.call(x86::rax);                                               // jit_stc(cpu, va, size, value)
      Label nobail = a.new_label();
      a.test(x86::eax, imm(0x100));                                   // 0x100 = translation-fault bail
      a.jz(nobail);
      set_pc(b->tag + 4 * (uint64_t) i);                              // resume at the faulting STx_C
      a.mov(x86::eax, imm(i));
      a.add(x86::eax, x86::r14d);
      a.jmp(done);
      a.bind(nobail);
      a.mov(reg(ra), x86::rax);                                       // Ra = success(1) / fail(0)
      continue;
    }

    // HW_LD physical (PALmode func 0/1): Ra = phys[Rb + disp12], no translation. jit_read_phys
    // does the aligned DRAM read (or replays in verify, bails on MMIO so the interpreter does the
    // ordered device read). disp is 12-bit here, not the 16-bit memory-format displacement.
    if (op == OP_HW_LDL || op == OP_HW_LDQ || op == OP_HW_LDQ_VPTE) {
      if (ra == 31) continue;                                  // R31 dest discards the read
      const int disp = (int) ((int32_t) (ins << 20) >> 20);    // sign-extend 12-bit displacement
      const int size_bits = (op == OP_HW_LDL) ? 32 : 64;
      if (rb == 31)  a.mov(x86::rdx, imm(disp));               // address (phys, or virtual for VPTE) -> RDX
      else        {  a.mov(x86::rdx, reg(rb)); if (disp) a.add(x86::rdx, imm(disp)); }
      a.mov(x86::rcx, x86::rsi);                               // cpu
      a.mov(x86::r8d, imm(size_bits));                         // size in bits
      a.lea(x86::r9, x86::qword_ptr(x86::rsp, 32));            // &out slot
      // func 5 -> jit_read_vpte (kernel-checked virtual read); else jit_read_phys
      a.mov(x86::rax, imm((uint64_t) (op == OP_HW_LDQ_VPTE ? read_vpte_helper : hw_ld_helper)));
      a.call(x86::rax);
      Label ok = a.new_label();
      a.test(x86::eax, x86::eax);
      a.jz(ok);
      set_pc(b->tag + 4 * (uint64_t) i);                       // resume at the faulting HW_LD
      a.mov(x86::eax, imm(i));                                  // this iteration: i instrs done
      a.add(x86::eax, x86::r14d);                               // + earlier chained iterations
      a.jmp(done);
      a.bind(ok);
      if (op == OP_HW_LDL) a.movsxd(x86::rax, x86::dword_ptr(x86::rsp, 32));
      else                 a.mov(x86::rax, x86::qword_ptr(x86::rsp, 32));   // HW_LDQ / VPTE: full quad
      a.mov(reg(ra), x86::rax);
      continue;
    }

    // Load-locked LDL_L/LDQ_L (0x2a/0x2b): Ra = MEM[Rb + disp16] AND establish the LL/SC exclusive
    // monitor. 
    if (op == OP_LDL_L || op == OP_LDQ_L) {
      const int disp = (int) (int16_t) (ins & 0xFFFF);
      const int size_bits = (op == OP_LDQ_L) ? 64 : 32;
      if (rb == 31)  a.mov(x86::rdx, imm(disp));               // va -> RDX
      else        {  a.mov(x86::rdx, reg(rb)); if (disp) a.add(x86::rdx, imm(disp)); }
      a.mov(x86::rcx, x86::rsi);                               // cpu
      a.mov(x86::r8d, imm(size_bits));                         // size in bits
      a.lea(x86::r9, x86::qword_ptr(x86::rsp, 32));            // &out slot
      a.mov(x86::rax, imm((uint64_t) read_locked_helper));
      a.call(x86::rax);                                        // jit_read_locked(cpu, va, size, &out)
      Label ok = a.new_label();
      a.test(x86::eax, x86::eax);
      a.jz(ok);
      set_pc(b->tag + 4 * (uint64_t) i);                       // resume at the faulting LDx_L
      a.mov(x86::eax, imm(i));
      a.add(x86::eax, x86::r14d);
      a.jmp(done);
      a.bind(ok);
      a.mov(x86::rax, x86::qword_ptr(x86::rsp, 32));           // *out already sign-extended by the helper
      a.mov(reg(ra), x86::rax);
      continue;
    }

    // HW_MTPR (PALmode, side-effect-free IPRs): jit_hw_mtpr(cpu, function, value=Rb) stores an IPR
    // field directly. The verify pass snapshots+compares those live-state writes. No fault/bail.
    if (op == OP_HW_MTPR) {
      const uint32_t function = (ins >> 8) & 0xff;
      a.mov(x86::rcx, x86::rsi);                               // cpu
      a.mov(x86::edx, imm(function));                          // IPR function number
      if (rb == 31)  a.xor_(x86::r8d, x86::r8d);               // value -> R8 (R31 == 0)
      else           a.mov(x86::r8, reg(rb));
      a.mov(x86::rax, imm((uint64_t) hw_mtpr_helper));
      a.call(x86::rax);                                        // jit_hw_mtpr(cpu, function, value)
      continue;
    }

    // HW_ST physical (PALmode func 0/1): phys[Rb + disp12] = Ra, no translation. jit_write_phys
    // does the aligned DRAM write (or compares the logged store in verify, bails on MMIO). disp is
    // 12-bit here, not the 16-bit memory-format displacement.
    if (op == OP_HW_STL || op == OP_HW_STQ) {
      const int disp = (int) ((int32_t) (ins << 20) >> 20);    // sign-extend 12-bit displacement
      const int size_bits = (op == OP_HW_STQ) ? 64 : 32;
      if (rb == 31)  a.mov(x86::rdx, imm(disp));               // phys addr -> RDX
      else        {  a.mov(x86::rdx, reg(rb)); if (disp) a.add(x86::rdx, imm(disp)); }
      if (ra == 31)  a.xor_(x86::r9d, x86::r9d);               // value -> R9 (R31 == 0)
      else           a.mov(x86::r9, reg(ra));
      a.mov(x86::rcx, x86::rsi);                               // cpu
      a.mov(x86::r8d, imm(size_bits));                         // size in bits
      a.mov(x86::rax, imm((uint64_t) hw_st_helper));
      a.call(x86::rax);                                        // jit_write_phys(cpu, phys, size, value)
      Label ok = a.new_label();
      a.test(x86::eax, x86::eax);
      a.jz(ok);
      set_pc(b->tag + 4 * (uint64_t) i);                       // resume at the faulting HW_ST
      a.mov(x86::eax, imm(i));
      a.add(x86::eax, x86::r14d);
      a.jmp(done);
      a.bind(ok);
      continue;
    }

    // Load-address: Ra = Rb + disp16 (LDA) or Rb + (disp16 << 16) (LDAH). Pure register
    // arithmetic. R31 dest discards the result (a NOP).
    if (op == OP_LDA || op == OP_LDAH) {
      if (ra == 31) continue;
      int64_t d = (int64_t) (int16_t) (ins & 0xFFFF);
      if (op == OP_LDAH) d <<= 16;
      if (rb == 31)  a.mov(x86::rax, imm(d));
      else        {  a.mov(x86::rax, reg(rb)); if (d) a.add(x86::rax, imm(d)); }
      a.mov(reg(ra), x86::rax);
      continue;
    }

    // HW_MFPR (0x19, PALmode): read the IPR named by (ins>>8)&0xff into Ra. The helper is an
    // independent reimplementation of DO_HW_MFPR that RETURNS the value (it reads state only, never
    // writes it). pass the current Ra as `cur` so an unknown IPR returns it unchanged (matching interp),
    // and write reg(ra) here so the value lands in whichever regs[] array we hold. Every MFPR IPR is
    // a pure read
    if (op == OP_HW_MFPR) {
      if (ra != 31) {                                  // MFPR R31 discards the value (R31 is hardwired 0)
        a.mov(x86::rcx, x86::rsi);                     // cpu
        a.mov(x86::edx, imm(ins));                     // full instruction (helper extracts the IPR index)
        a.mov(x86::r8, reg(ra));                       // cur: current Ra, returned as-is for an unknown IPR
        a.mov(x86::rax, imm((uint64_t) hw_mfpr_helper));
        a.call(x86::rax);                              // -> RAX = IPR value
        a.mov(reg(ra), x86::rax);                      // Ra = value (reg() applies the PALshadow remap)
      }
      continue;
    }

    // MISC state reads RPCC/RC/RS (0x18): Ra = jit_misc(cpu, sel). The helper reads the cycle
    // counter / interrupt flag (clearing or setting it) in production, and replays the interp
    // pass's value in verify. Dest is Ra (not Rc); classify gated Ra!=31, so a store always happens.
    if (op == OP_RPCC || op == OP_RC || op == OP_RS) {
      const int sel = (op == OP_RPCC) ? 0 : (op == OP_RC) ? 1 : 2;
      a.mov(x86::rcx, x86::rsi);                       // cpu
      a.mov(x86::edx, imm(sel));                       // selector: 0=RPCC, 1=RC, 2=RS
      a.mov(x86::rax, imm((uint64_t) misc_helper));
      a.call(x86::rax);                                // -> RAX = value (replayed in verify)
      a.mov(reg(ra), x86::rax);                        // Ra = value (reg() applies the PALshadow remap)
      continue;
    }

    // ITOFx (0x14): f[Fc] = fmt(Ra). jit_itof mirrors FPSTART (fpen -> FEN trap = bail, exc_sum=0);
    // the verify compares the FP file via its snapshot.
    if (op == OP_ITOFS || op == OP_ITOFF || op == OP_ITOFT) {
      a.mov(x86::rcx, x86::rsi);                       // cpu
      a.mov(x86::edx, imm(rc));                        // Fc (plain index, FREG has no shadow remap)
      if (ra == 31)  a.xor_(x86::r8d, x86::r8d);       // value (R31 == 0)
      else           a.mov(x86::r8, reg(ra));
      a.mov(x86::r9d, imm(op == OP_ITOFS ? 1 : (op == OP_ITOFF) ? 2 : 0));   // fmt: S/F/T
      a.mov(x86::rax, imm((uint64_t) itof_helper));
      a.call(x86::rax);                                // jit_itof(cpu, fc, value, fmt)
      Label ok = a.new_label();
      a.test(x86::eax, x86::eax);
      a.jz(ok);
      set_pc(b->tag + 4 * (uint64_t) i);               // FEN trap: resume here in the interpreter
      a.mov(x86::eax, imm(i));
      a.add(x86::eax, x86::r14d);
      a.jmp(done);
      a.bind(ok);
      continue;
    }

    // FTOIx (0x1c): Rc = fmt(f[Fa]). Same FPSTART bail shape; dest is a GPR (verify-compared).
    if (op == OP_FTOIS || op == OP_FTOIT) {
      a.mov(x86::rcx, x86::rsi);                       // cpu
      a.mov(x86::edx, imm(ra));                        // Fa (the Ra field, plain index)
      a.mov(x86::r8d, imm(op == OP_FTOIS ? 1 : 0));    // fmt: S/T
      a.lea(x86::r9, x86::qword_ptr(x86::rsp, 32));    // &out
      a.mov(x86::rax, imm((uint64_t) ftoi_helper));
      a.call(x86::rax);                                // jit_ftoi(cpu, fa, fmt, &out)
      Label ok = a.new_label();
      a.test(x86::eax, x86::eax);
      a.jz(ok);
      set_pc(b->tag + 4 * (uint64_t) i);               // FEN trap: resume here in the interpreter
      a.mov(x86::eax, imm(i));
      a.add(x86::eax, x86::r14d);
      a.jmp(done);
      a.bind(ok);
      a.mov(x86::rax, x86::qword_ptr(x86::rsp, 32));
      a.mov(reg(rc), x86::rax);                        // Rc (reg() applies the PALshadow remap)
      continue;
    }

    // FLTL non-arithmetic (0x17): all effects in state.f / fpcr via jit_fltl(cpu, ins).
    if (op == OP_FLTL) {
      a.mov(x86::rcx, x86::rsi);                       // cpu
      a.mov(x86::edx, imm(ins));                       // full instruction (helper decodes func/regs)
      a.mov(x86::rax, imm((uint64_t) fltl_helper));
      a.call(x86::rax);                                // jit_fltl(cpu, ins)
      Label ok = a.new_label();
      a.test(x86::eax, x86::eax);
      a.jz(ok);
      set_pc(b->tag + 4 * (uint64_t) i);               // FEN trap: resume here in the interpreter
      a.mov(x86::eax, imm(i));
      a.add(x86::eax, x86::r14d);
      a.jmp(done);
      a.bind(ok);
      continue;
    }

    // Computed jump JMP/JSR/RET (0x1a): Ra = PC+4 (return address); PC = Rb & ~3. A
    // terminator like a branch, but the target is a register -- left in R10 so the epilogue's
    // cached link validates it (succ->tag == target), chaining calls/returns/dispatch.
    if (op == OP_JMP) {
      const uint64_t ret = b->tag + 4 * (uint64_t) (i + 1);
      if (rb == 31) a.xor_(x86::r10d, x86::r10d);
      else          a.mov(x86::r10, reg(rb));
      a.and_(x86::r10, imm(~(uint64_t) 3));                       // target = Rb & ~3 (clear low 2)
      if (b->tag & 3) a.or_(x86::r10, imm(b->tag & 3));           // DO_JMP: mode bits come from the current pc
      if (ra != 31) { a.mov(x86::rax, imm(ret & ~(uint64_t) 3)); a.mov(reg(ra), x86::rax); }  // return addr = PC & ~3 (DO_JMP)
      a.mov(x86::qword_ptr(x86::rsi, m_off.state_pc), x86::r10);  // state.pc = target
      continue;
    }

    // HW_RET (HWREI, 0x1e): a PAL return -- a simple computed jump, target = Rb & ~2, no return-
    // address write. Like OP_JMP, leaves R10 = target for the epilogue's in-frame chain.
    if (op == OP_HW_RET) {
      if (rb == 31) a.xor_(x86::r10d, x86::r10d);
      else          a.mov(x86::r10, reg(rb));
      a.and_(x86::r10, imm(~(uint64_t) 2));                       // target = Rb & ~2 (clear bit 1)
      a.mov(x86::qword_ptr(x86::rsi, m_off.state_pc), x86::r10);  // state.pc = target
      continue;
    }

    // CALL_PAL (0x00): vector to the PALcode entry, saving the return address in R23 and the
    // faulting PC in EXC_ADDR (per ENTER_NATIVE_CALL_PAL). 
    if (op == OP_CALL_PAL) {
      const uint32_t func = ins & 0x1FFFFFFF;
      const uint64_t cpc  = b->tag + 4 * (uint64_t) i;                          // CALL_PAL address
      const uint64_t ret  = (b->tag + 4 * (uint64_t) (i + 1)) & ~(uint64_t) 2;  // return addr (PC & ~2)
      const uint64_t voff = (uint64_t) 0x2000 | ((uint64_t) (func & 0x80) << 5)
                          | ((uint64_t) (func & 0x3f) << 6) | (uint64_t) 1;     // PAL entry offset
      Label do_vector = a.new_label();
      if (func < 0x40) {                          // privileged: OPCDEC trap if in user mode (cm != 0)
        a.cmp(x86::dword_ptr(x86::rsi, m_off.state_cm), imm(0));
        a.je(do_vector);
        a.mov(x86::rcx, x86::rsi);                 // cpu
        a.mov(x86::rdx, imm(cpc));                 // faulting PC (-> EXC_ADDR)
        a.mov(x86::rax, imm((uint64_t) opcdec_helper));
        a.call(x86::rax);                          // jit_opcdec: sets state.pc/exc_addr, clears lock
        a.add(x86::r14d, imm(plen));               // count the block; helper already wrote state.pc
        a.mov(x86::eax, x86::r14d);
        a.jmp(done);                               // trap path exits (does not chain)
      }
      a.bind(do_vector);
      a.mov(x86::rax, imm(cpc));                                  // EXC_ADDR = CALL_PAL address
      a.mov(x86::qword_ptr(x86::rsi, m_off.exc_addr), x86::rax);
      a.movzx(x86::eax, x86::byte_ptr(x86::rsi, m_off.sde));      // SDE (0/1)
      a.shl(x86::eax, imm(5));                                    // * 32
      a.add(x86::eax, imm(23));                                   // R23 index: 23, or 55 if SDE
      a.mov(x86::rcx, imm(ret));
      a.mov(x86::qword_ptr(x86::rbx, x86::rax, 3), x86::rcx);     // r[idx] = return address
      a.mov(x86::r10, x86::qword_ptr(x86::rsi, m_off.pal_base));
      a.or_(x86::r10, imm(voff));                                 // r10 = pal_base | entry offset
      a.mov(x86::qword_ptr(x86::rsi, m_off.state_pc), x86::r10);  // state.pc = PAL entry
      continue;                                                  // -> terminator epilogue chains via r10
    }

    // Branch terminators: compute the target into state.pc, then end the block. The
    // branch is at index i, so the PC of the next instruction is b->tag + 4*(i+1).
    if (is_branch(op)) {
      const int64_t  bdisp = (int64_t) ((uint64_t) (ins & 0x1FFFFF) << 43) >> 43;  // sext disp21
      const uint64_t fall  = b->tag + 4 * (uint64_t) (i + 1);
      const uint64_t tgt   = fall + (uint64_t) (bdisp * 4);
      if (op == OP_BR || op == OP_BSR) {                 // Ra = return address; PC = target
        if (ra != 31) { a.mov(x86::r10, imm(fall & ~(uint64_t) 3)); a.mov(reg(ra), x86::r10); }  // link = PC & ~3 (DO_BR)
        a.mov(x86::r10, imm(tgt));
      } else {                                           // conditional: PC = cond ? target : fall
        if (ra == 31) a.xor_(x86::eax, x86::eax);
        else          a.mov(x86::rax, reg(ra));
        a.mov(x86::r10, imm(fall));
        a.mov(x86::r11, imm(tgt));
        if (op == OP_BLBC || op == OP_BLBS) a.test(x86::rax, imm(1));
        else                                a.test(x86::rax, x86::rax);
        switch (op) {
          case OP_BEQ:  a.cmovz(x86::r10, x86::r11);  break;
          case OP_BNE:  a.cmovnz(x86::r10, x86::r11); break;
          case OP_BLT:  a.cmovs(x86::r10, x86::r11);  break;
          case OP_BGE:  a.cmovns(x86::r10, x86::r11); break;
          case OP_BLE:  a.cmovle(x86::r10, x86::r11); break;
          case OP_BGT:  a.cmovg(x86::r10, x86::r11);  break;
          case OP_BLBC: a.cmovz(x86::r10, x86::r11);  break;
          case OP_BLBS: a.cmovnz(x86::r10, x86::r11); break;
          default: break;
        }
      }
      a.mov(x86::qword_ptr(x86::rsi, m_off.state_pc), x86::r10);
      continue;
    }

    // INTL conditional moves (CMOVxx): Rc = cond(Ra) ? op2 : Rc. Same condition tests as the
    // matching branches; op2 (Rb or literal) moves into Rc only when the condition holds, so the
    // current Rc is read and kept otherwise. R31 dest discards the result.
    if (op == OP_CMOV) {
      if (rc == 31) continue;
      const uint32_t f = (ins >> 5) & 0x7f;
      op1_rax();                                  // rax = Ra (condition operand)
      op2_rcx();                                  // rcx = op2 (Rb or literal -- the moved value)
      a.mov(x86::r10, reg(rc));                   // r10 = current Rc (kept when the condition fails)
      if (f == 0x14 || f == 0x16) a.test(x86::rax, imm(1));    // CMOVLBS/LBC: test the low bit
      else                        a.test(x86::rax, x86::rax);  // others: test the full value
      switch (f) {
        case 0x24: a.cmovz(x86::r10, x86::rcx);  break;   // CMOVEQ  (Ra == 0)
        case 0x26: a.cmovnz(x86::r10, x86::rcx); break;   // CMOVNE  (Ra != 0)
        case 0x44: a.cmovs(x86::r10, x86::rcx);  break;   // CMOVLT  (Ra <  0)
        case 0x46: a.cmovns(x86::r10, x86::rcx); break;   // CMOVGE  (Ra >= 0)
        case 0x64: a.cmovle(x86::r10, x86::rcx); break;   // CMOVLE  (Ra <= 0)
        case 0x66: a.cmovg(x86::r10, x86::rcx);  break;   // CMOVGT  (Ra >  0)
        case 0x14: a.cmovnz(x86::r10, x86::rcx); break;   // CMOVLBS (Ra & 1)
        case 0x16: a.cmovz(x86::r10, x86::rcx);  break;   // CMOVLBC (!(Ra & 1))
      }
      a.mov(reg(rc), x86::r10);
      continue;
    }

    // INTS (0x12) byte-manipulation: extract/insert/mask/zap. Rc = a shift+mask of Ra keyed on the
    // byte position pos = Rb&7 (the selector V_2). Pure ALU -> verify-checked by the GPR compare.
    // RAX = Ra/result, CL = the variable shift, R10/R11 scratch, EDX preserves pos for the H-form
    // pos==0 edge cases. Mirrors cpu_bwx.h; size (0=B 1=W 2=L 3=Q) and mask come from the function.
    if (op == OP_EXTL || op == OP_EXTH || op == OP_INSL || op == OP_INSH
        || op == OP_MSKL || op == OP_MSKH || op == OP_ZAP) {
      if (rc == 31) continue;
      const uint32_t f = (ins >> 5) & 0x7f;
      const int size = (f >> 4) & 3;
      const uint64_t mask = (size == 0) ? (uint64_t) 0xff : (size == 1) ? (uint64_t) 0xffff
                          : (size == 2) ? (uint64_t) 0xffffffff : ~(uint64_t) 0;
      op1_rax();                                                  // rax = Ra (data)
      op2_rcx();                                                  // rcx = Rb / literal (selector)
      switch (op) {
        case OP_EXTL:                                             // (Ra >> pos*8) & mask
          a.and_(x86::ecx, imm(7)); a.shl(x86::ecx, imm(3));
          a.shr(x86::rax, x86::cl);
          if (size != 3) { a.mov(x86::r10, imm(mask)); a.and_(x86::rax, x86::r10); }
          break;
        case OP_EXTH:                                             // (Ra << ((64-pos*8)&63)) & mask
          a.and_(x86::ecx, imm(7)); a.shl(x86::ecx, imm(3));
          a.neg(x86::ecx); a.and_(x86::ecx, imm(63));
          a.shl(x86::rax, x86::cl);
          if (size != 3) { a.mov(x86::r10, imm(mask)); a.and_(x86::rax, x86::r10); }
          break;
        case OP_INSL:                                             // (Ra & mask) << pos*8
          if (size != 3) { a.mov(x86::r10, imm(mask)); a.and_(x86::rax, x86::r10); }
          a.and_(x86::ecx, imm(7)); a.shl(x86::ecx, imm(3));
          a.shl(x86::rax, x86::cl);
          break;
        case OP_INSH:                                             // pos ? ((Ra&mask) >> ((64-pos*8)&63)) : 0
          if (size != 3) { a.mov(x86::r10, imm(mask)); a.and_(x86::rax, x86::r10); }
          a.and_(x86::ecx, imm(7)); a.mov(x86::edx, x86::ecx);
          a.shl(x86::ecx, imm(3)); a.neg(x86::ecx); a.and_(x86::ecx, imm(63));
          a.shr(x86::rax, x86::cl);
          a.xor_(x86::r11d, x86::r11d); a.test(x86::edx, x86::edx); a.cmovz(x86::rax, x86::r11);
          break;
        case OP_MSKL:                                             // Ra & ~(mask << pos*8)
          a.and_(x86::ecx, imm(7)); a.shl(x86::ecx, imm(3));
          a.mov(x86::r10, imm(mask)); a.shl(x86::r10, x86::cl);
          a.not_(x86::r10); a.and_(x86::rax, x86::r10);
          break;
        case OP_MSKH:                                             // pos ? (Ra & ~(mask >> ((64-pos*8)&63))) : Ra
          a.and_(x86::ecx, imm(7)); a.mov(x86::edx, x86::ecx);
          a.shl(x86::ecx, imm(3)); a.neg(x86::ecx); a.and_(x86::ecx, imm(63));
          a.mov(x86::r10, imm(mask)); a.shr(x86::r10, x86::cl); a.not_(x86::r10);
          a.mov(x86::r11, x86::rax); a.and_(x86::r11, x86::r10);
          a.test(x86::edx, x86::edx); a.cmovnz(x86::rax, x86::r11);
          break;
        case OP_ZAP:                                              // Ra & byte_expand(selector); ZAP inverts
          a.movzx(x86::ecx, x86::cl);
          a.mov(x86::r11, imm((uint64_t) &g_zapnot_mask[0]));
          a.mov(x86::r10, x86::qword_ptr(x86::r11, x86::rcx, 3));   // g_zapnot_mask[selector & 0xff]
          if (f == 0x30) a.not_(x86::r10);                        // ZAP keeps bytes whose bit is CLEAR
          a.and_(x86::rax, x86::r10);
          break;
        default: break;
      }
      a.mov(reg(rc), x86::rax);
      continue;
    }

    switch (op) {
      case OP_ADDQ:  op1_rax(); op2_rcx(); a.add(x86::rax, x86::rcx); break;
      case OP_SUBQ:  op1_rax(); op2_rcx(); a.sub(x86::rax, x86::rcx); break;
      case OP_AND:   op1_rax(); op2_rcx(); a.and_(x86::rax, x86::rcx); break;
      case OP_BIS:   op1_rax(); op2_rcx(); a.or_(x86::rax, x86::rcx); break;
      case OP_XOR:   op1_rax(); op2_rcx(); a.xor_(x86::rax, x86::rcx); break;
      case OP_BIC:   op1_rax(); op2_rcx(); a.not_(x86::rcx); a.and_(x86::rax, x86::rcx); break;
      case OP_ORNOT: op1_rax(); op2_rcx(); a.not_(x86::rcx); a.or_(x86::rax, x86::rcx); break;
      case OP_EQV:   op1_rax(); op2_rcx(); a.not_(x86::rcx); a.xor_(x86::rax, x86::rcx); break;
      case OP_MULQ:  op1_rax(); op2_rcx(); a.imul(x86::rax, x86::rcx); break;
      case OP_UMULH: op1_rax(); op2_rcx(); a.mul(x86::rcx); a.mov(x86::rax, x86::rdx); break;  // RDX:RAX = Ra*op2; hi64 = RDX
      case OP_MULL:                                          // 32-bit multiply, low 32 sign-extended
      {
        if (ra == 31) a.xor_(x86::eax, x86::eax);
        else          a.mov(x86::eax, reg32(ra));
        if (islit)         a.mov(x86::ecx, imm(lit));
        else if (rb == 31) a.xor_(x86::ecx, x86::ecx);
        else               a.mov(x86::ecx, reg32(rb));
        a.imul(x86::eax, x86::ecx);                          // eax = (Ra*op2)[31:0]
        a.movsxd(x86::rax, x86::eax);
        break;
      }

      case OP_S4ADDQ: op1_rax(); a.shl(x86::rax, imm(2)); op2_rcx(); a.add(x86::rax, x86::rcx); break;
      case OP_S8ADDQ: op1_rax(); a.shl(x86::rax, imm(3)); op2_rcx(); a.add(x86::rax, x86::rcx); break;
      case OP_S4SUBQ: op1_rax(); a.shl(x86::rax, imm(2)); op2_rcx(); a.sub(x86::rax, x86::rcx); break;
      case OP_S8SUBQ: op1_rax(); a.shl(x86::rax, imm(3)); op2_rcx(); a.sub(x86::rax, x86::rcx); break;

      case OP_SLL: op1_rax(); op2_rcx(); a.shl(x86::rax, x86::cl); break;
      case OP_SRL: op1_rax(); op2_rcx(); a.shr(x86::rax, x86::cl); break;
      case OP_SRA: op1_rax(); op2_rcx(); a.sar(x86::rax, x86::cl); break;

      case OP_SEXTB: op2_rcx(); a.movsx(x86::rax, x86::cl); break;   // Rc = sign-extend low byte of op2 (V_2)
      case OP_SEXTW: op2_rcx(); a.movsx(x86::rax, x86::cx); break;   // Rc = sign-extend low word of op2 (V_2)

      case OP_AMASK:    // Rc = op2 & ~CPU_AMASK -- EV68 feature mask 0x1307 (keep in sync w/ cpu_defs.h);
        op2_rcx();      // classify enforced Ra==31 (the Ra!=31 form traps OPCDEC in the interpreter)
        a.mov(x86::rax, imm(~(uint64_t) 0x1307));
        a.and_(x86::rax, x86::rcx);
        break;
      case OP_IMPLVER:  // Rc = CPU_IMPLVER (2 = EV6 family; keep in sync w/ cpu_defs.h)
        a.mov(x86::eax, imm(2));
        break;

      case OP_CMPEQ:  op1_rax(); op2_rcx(); a.cmp(x86::rax, x86::rcx); a.sete(x86::al);  a.movzx(x86::eax, x86::al); break;
      case OP_CMPLT:  op1_rax(); op2_rcx(); a.cmp(x86::rax, x86::rcx); a.setl(x86::al);  a.movzx(x86::eax, x86::al); break;
      case OP_CMPLE:  op1_rax(); op2_rcx(); a.cmp(x86::rax, x86::rcx); a.setle(x86::al); a.movzx(x86::eax, x86::al); break;
      case OP_CMPULT: op1_rax(); op2_rcx(); a.cmp(x86::rax, x86::rcx); a.setb(x86::al);  a.movzx(x86::eax, x86::al); break;
      case OP_CMPULE: op1_rax(); op2_rcx(); a.cmp(x86::rax, x86::rcx); a.setbe(x86::al); a.movzx(x86::eax, x86::al); break;

      case OP_CMPBGE:   // 8 parallel unsigned byte compares (Ra.byte[i] >= op2.byte[i]) -> bit i; bits 63:8 = 0
        op1_rax(); op2_rcx();
        a.movq(x86::xmm0, x86::rax);             // Ra (8 bytes)
        a.movq(x86::xmm1, x86::rcx);             // op2 (8 bytes; literal in byte 0, 0 elsewhere)
        a.movdqa(x86::xmm2, x86::xmm0);          // copy of Ra
        a.pmaxub(x86::xmm2, x86::xmm1);          // per-byte unsigned max(Ra, op2)
        a.pcmpeqb(x86::xmm2, x86::xmm0);         // 0xff where max == Ra, i.e. Ra >= op2
        a.pmovmskb(x86::eax, x86::xmm2);         // gather byte-sign bits -> low 8 bits = the mask
        a.movzx(x86::eax, x86::al);              // discard the high (zero-padded) lane bits
        break;

      case OP_ADDL: case OP_SUBL:                                    // 32-bit, result sign-extended to 64
      case OP_S4ADDL: case OP_S8ADDL: case OP_S4SUBL: case OP_S8SUBL: // scaled longword: sext32((Ra*scale) +/- Rb)
      {
        const bool issub = (op == OP_SUBL || op == OP_S4SUBL || op == OP_S8SUBL);
        const int  sh    = (op == OP_S4ADDL || op == OP_S4SUBL) ? 2     // Ra*4
                         : (op == OP_S8ADDL || op == OP_S8SUBL) ? 3     // Ra*8
                         : 0;
        if (ra == 31) a.xor_(x86::eax, x86::eax);
        else          a.mov(x86::eax, reg32(ra));   // shadow-remapped (was a raw rbx read)
        if (sh) a.shl(x86::eax, imm(sh));            // scale in 32-bit: (Ra<<sh)[31:0] == ((RAV<<sh)+..)[31:0]
        if (islit)         a.mov(x86::ecx, imm(lit));
        else if (rb == 31) a.xor_(x86::ecx, x86::ecx);
        else               a.mov(x86::ecx, reg32(rb));
        if (issub) a.sub(x86::eax, x86::ecx);
        else       a.add(x86::eax, x86::ecx);
        a.movsxd(x86::rax, x86::eax);
        break;
      }
      default: break;
    }

    if (rc != 31) a.mov(reg(rc), x86::rax);
  }
  // Epilogue. Count this block's instructions, then chain into the next block (staying
  // in native code) or return to the dispatcher.
#ifndef JIT_VERIFY
  // Gate the chain: stop if we've hit the budget ceiling or an interrupt/timer is pending
  auto emit_gate = [&](Label& lbl) {
    a.cmp(x86::r14, x86::qword_ptr(x86::rsi, m_off.jit_budget)); a.jge(lbl);
    a.cmp(x86::byte_ptr(x86::rsi, m_off.check_int), imm(0));     a.jne(lbl);
    a.cmp(x86::byte_ptr(x86::rsi, m_off.check_timers), imm(0));  a.jne(lbl);
  };
  // Field offsets within a JitBlock, so the epilogue can validate a cached successor via
  // [succ + off]. 
  const uint32_t off_body = (uint32_t) ((char*) &b->jit_body  - (char*) b);
  const uint32_t off_tag  = (uint32_t) ((char*) &b->tag       - (char*) b);
  const uint32_t off_vgen = (uint32_t) ((char*) &b->vgen      - (char*) b);
  // Cached direct link: tail straight into our cached successor's body if it's still live
  // -- compiled (jit_body != 0, cleared on flush/recompile) AND still maps this exit's PC
  // (tag == R10) AND, for native targets, phys-validated under the current ITB generation
  // (gen-stale -> dispatcher, which rechecks phys and re-stamps). Otherwise record a patch
  // request (m_link_from = this block) and fall back. R10 = next PC; clobbers RAX/RCX/RDX.
  auto emit_chain = [&](Label& lbl) {
    Label miss = a.new_label();
    a.mov(x86::rax, imm((uint64_t) &b->link));
    a.mov(x86::rax, x86::qword_ptr(x86::rax));                 // succ = b->link
    a.test(x86::rax, x86::rax);                            a.jz(miss);
    a.mov(x86::rcx, x86::qword_ptr(x86::rax, off_body));       // succ->jit_body
    a.test(x86::rcx, x86::rcx);                            a.jz(miss);
    a.mov(x86::rdx, x86::qword_ptr(x86::rax, off_tag));        // succ->tag
    a.cmp(x86::rdx, x86::r10);                             a.jne(miss);   // not this exit's target
    // Single epoch guard, all successors: vgen = itb_gen + flush_gen at last validation; both
    // counters are monotonic, so one sum compare catches a remap OR a flush since then.
    const uint32_t fg_off = (uint32_t) ((char*) &m_flush_gen - (char*) &m_itb_gen);
    a.mov(x86::rdx, imm((uint64_t) &m_itb_gen));
    a.mov(x86::r11, x86::qword_ptr(x86::rdx));
    a.add(x86::r11, x86::qword_ptr(x86::rdx, fg_off));         // current epoch sum
    a.cmp(x86::qword_ptr(x86::rax, off_vgen), x86::r11);  a.jne(miss);   // stale: revalidate via dispatcher
    // PALmode successor (tag bit 0): its shadow remap assumes SDE -- the dispatcher's guard.
    Label chain_ok = a.new_label();
    a.test(x86::r10, imm(1));                             a.jz(chain_ok);
    a.cmp(x86::byte_ptr(x86::rsi, m_off.sde), imm(0));    a.je(miss);    // PALmode + !SDE: don't
    a.bind(chain_ok);
    a.jmp(x86::rcx);                                           // HIT: tail in (shared frame)
    a.bind(miss);
    a.mov(x86::rax, imm((uint64_t) b));
    a.mov(x86::qword_ptr(x86::rsi, m_off.link_from), x86::rax);   // request b->link patch
    // fall through to lbl (return to dispatcher)
  };
#endif
  if (terminator_jmp) {
    // Computed jump (JMP / HW_RET): R10 holds the register target (already written to state.pc).
    // Chain in-frame via jit_indirect -- the dispatcher's own block-cache lookup -- tailing into
    // the target's compiled body when it's live + runnable here. Unlike the old single-slot link
    // this keys on the ACTUAL target, so it handles all targets with no thrash on varying jumps.
    a.add(x86::r14d, imm(plen));
#ifndef JIT_VERIFY
    Label exit_chain = a.new_label();
    emit_gate(exit_chain);                                        // budget/interrupt: bail to dispatcher
    a.mov(x86::rcx, x86::rsi);                                    // cpu
    a.mov(x86::rdx, x86::r10);                                    // target PC (== state.pc)
    a.mov(x86::rax, imm((uint64_t) indirect_helper));
    a.call(x86::rax);                                             // jit_indirect(cpu, target) -> body | 0
    a.test(x86::rax, x86::rax);                              a.jz(exit_chain);
    a.jmp(x86::rax);                                              // HIT: tail into the target's body
    a.bind(exit_chain);
#endif
  } else if (terminator_branch) {
    a.add(x86::r14d, imm(plen));   // R10 still holds the next PC (branch wrote state.pc + R10)
#ifndef JIT_VERIFY
    Label exit_chain = a.new_label();
    Label not_self   = a.new_label();
    emit_gate(exit_chain);
    // Self-loop fast path: a taken branch back to our own start (r10 == b->tag) jumps
    // straight into the body, skipping the resolver call entirely.
    a.mov(x86::rax, imm(b->tag));                               // tag may exceed imm32
    a.cmp(x86::r10, x86::rax);                                  a.jne(not_self);
    a.jmp(body);
    a.bind(not_self);
    emit_chain(exit_chain);
    a.bind(exit_chain);
#endif
  } else {
    set_pc(b->tag + 4 * (uint64_t) plen);   // straight-line fall-through to the next block
    a.add(x86::r14d, imm(plen));
#ifndef JIT_VERIFY
    Label exit_chain = a.new_label();
    emit_gate(exit_chain);
    emit_chain(exit_chain);
    a.bind(exit_chain);
#endif
  }
  a.mov(x86::eax, x86::r14d);   // total instructions completed across the chain
  a.bind(done);                 // bail jumps here with EAX already set
  a.add(x86::rsp, imm(48));
  a.pop(x86::r14);
  a.pop(x86::rsi);
  a.pop(x86::rbx);
  a.ret();

  const size_t csz = code.code_size();
  JitFn fn = nullptr;
  if (((JitRuntime*) m_rt)->add(&fn, &code) != Error::kOk) return;
  b->code = fn;
  b->jit_body = (void*) ((uint8_t*) (void*) fn + body_off);   // chained re-entry (past prologue)
  b->body_off = (uint32_t) body_off;                          // to restore jit_body on revalidate
  b->src_sum  = src_hash(dram + phys, b->n_instr);            // source fingerprint (revalidate vs self-mod)
  b->hash_len = b->n_instr;                                   // freeze the hash extent (n_instr drifts)
  b->prefix_len = plen;
  m_code_bytes += csz;   // track for the reclaim threshold (see flush())
#ifdef JIT_STATS
  m_stat_compiled++;
  m_stat_plen_sum += plen;
#endif
}

#ifdef JIT_VERIFY
void CJitEngine::verify_compare(uint64_t blk_virt, const uint64_t* interp, const uint64_t* jit,
                                const uint32_t* words, uint32_t nwords)
{
  m_v_exec++;
  for (int r = 0; r < 63; ++r) {   // r0..r30 main bank + r32..r62 PALshadow bank
    if (r == 31) continue;         // zero register
    if (interp[r] != jit[r]) {
      m_v_fail++;
      printf("[JIT][VERIFY] MISMATCH at block pc=%016llx: r%d interp=%016llx jit=%016llx\n",
             (unsigned long long) blk_virt, r,
             (unsigned long long) interp[r], (unsigned long long) jit[r]);
      printf("   words:");   // the compiled prefix, to decode which instruction diverged
      for (uint32_t w = 0; w < nwords && w < 16; ++w) printf(" %08x", words[w]);
      printf("\n");
      break;
    }
  }
  if ((m_v_exec % 500000) == 0)
    printf("[JIT][VERIFY] %llu compiled-block execs, %llu mismatches\n",
           (unsigned long long) m_v_exec, (unsigned long long) m_v_fail);
}
#endif

#ifdef JIT_STATS
// Short mnemonic for the opcode that ended a block's compiled prefix (the coverage gap).
static const char* opcode_name(unsigned op)
{
  switch (op) {
    case 0x00: return "CALL_PAL";
    case 0x08: return "LDA";   case 0x09: return "LDAH"; case 0x0a: return "LDBU";  case 0x0b: return "LDQ_U";
    case 0x0c: return "LDWU";  case 0x0d: return "STW";  case 0x0e: return "STB";   case 0x0f: return "STQ_U";
    case 0x10: return "INTA";  case 0x11: return "INTL"; case 0x12: return "INTS";  case 0x13: return "INTM";
    case 0x14: return "ITFP";  case 0x15: return "FLTV"; case 0x16: return "FLTI";  case 0x17: return "FLTL";
    case 0x18: return "MISC";  case 0x19: return "HWMFPR";case 0x1a: return "JMP";   case 0x1b: return "HWLD";
    case 0x1c: return "FPTI";  case 0x1d: return "HWMTPR";case 0x1e: return "HWREI"; case 0x1f: return "HWST";
    case 0x20: return "LDF";   case 0x21: return "LDG";  case 0x22: return "LDS";   case 0x23: return "LDT";
    case 0x24: return "STF";   case 0x25: return "STG";  case 0x26: return "STS";   case 0x27: return "STT";
    case 0x2a: return "LDL_L"; case 0x2b: return "LDQ_L";case 0x2e: return "STL_C"; case 0x2f: return "STQ_C";
    default:   return "op";
  }
}

void CJitEngine::note_exec(uint32_t native_instr, uint32_t interp_instr)
{
  m_stat_native += native_instr;
  m_stat_interp += interp_instr;
  if (native_instr) m_stat_hot++;     // one compiled-chain dispatch
  if (interp_instr) m_stat_miss++;    // one interpreted (cold/uncompilable) block
  const uint64_t total = m_stat_native + m_stat_interp;
  if (total < 100000000) return;      // report every 100M instructions

  // Build each line in a buffer and print it with ONE call: 4 CPU threads print concurrently,
  // and per-item printf loops interleave mid-line. The [CPU%d] tag attributes each line.
  const double chain = m_stat_hot ? (double) m_stat_native / (double) m_stat_hot : 0.0;
  const double avgpl = m_stat_compiled ? (double) m_stat_plen_sum / (double) m_stat_compiled : 0.0;
  char buf[512];
  int  len;
  printf("[JIT][STATS][CPU%d] native %.1f%% (%llu/%llu) | chain avg %.1f instr over %llu dispatches | interp %llu blks\n",
         m_cpu_id, 100.0 * (double) m_stat_native / (double) total,
         (unsigned long long) m_stat_native, (unsigned long long) total,
         chain, (unsigned long long) m_stat_hot, (unsigned long long) m_stat_miss);
  len = snprintf(buf, sizeof(buf), "[JIT][STATS][CPU%d] %llu recorded, %llu compiled (avg %.1f instr) | block-breakers:",
                 m_cpu_id, (unsigned long long) m_recorded, (unsigned long long) m_stat_compiled, avgpl);
  uint64_t hist[64];
  memcpy(hist, m_term_op, sizeof(hist));
  for (int rank = 0; rank < 8 && len < (int) sizeof(buf) - 32; ++rank) {
    int best = -1; uint64_t bestv = 0;
    for (int op = 0; op < 64; ++op) if (hist[op] > bestv) { bestv = hist[op]; best = op; }
    if (best < 0) break;
    len += snprintf(buf + len, sizeof(buf) - len, " %s(0x%02x)=%llu", opcode_name(best), best, (unsigned long long) bestv);
    hist[best] = 0;
  }
  printf("%s\n", buf);
  if (m_term_op[0]) {   // CALL_PAL cut blocks -- show which function codes dominate (chain targets)
    len = snprintf(buf, sizeof(buf), "[JIT][STATS][CPU%d]   CALL_PAL by func:", m_cpu_id);
    uint64_t ph[256];
    memcpy(ph, m_pal_func, sizeof(ph));
    for (int rank = 0; rank < 8 && len < (int) sizeof(buf) - 32; ++rank) {
      int best = -1; uint64_t bestv = 0;
      for (int f = 0; f < 256; ++f) if (ph[f] > bestv) { bestv = ph[f]; best = f; }
      if (best < 0) break;
      len += snprintf(buf + len, sizeof(buf) - len, " 0x%02x=%llu", best, (unsigned long long) bestv);
      ph[best] = 0;
    }
    printf("%s\n", buf);
  }
  if (m_term_op[0x1d]) {   // HW_MTPR cut blocks -- which IPR writes dominate (the side-effecting set)
    len = snprintf(buf, sizeof(buf), "[JIT][STATS][CPU%d]   HW_MTPR by IPR:", m_cpu_id);
    uint64_t mh[256];
    memcpy(mh, m_mtpr_func, sizeof(mh));
    for (int rank = 0; rank < 8 && len < (int) sizeof(buf) - 32; ++rank) {
      int best = -1; uint64_t bestv = 0;
      for (int f = 0; f < 256; ++f) if (mh[f] > bestv) { bestv = mh[f]; best = f; }
      if (best < 0) break;
      len += snprintf(buf + len, sizeof(buf) - len, " 0x%02x=%llu", best, (unsigned long long) bestv);
      mh[best] = 0;
    }
    printf("%s\n", buf);
  }
  if (m_term_op[0x1b]) {   // HW_LD cut blocks -- which forms dominate (virt/lock/vpte/chk)
    len = snprintf(buf, sizeof(buf), "[JIT][STATS][CPU%d]   HW_LD by form:", m_cpu_id);
    uint64_t lh[16];
    memcpy(lh, m_hwld_func, sizeof(lh));
    for (int rank = 0; rank < 8 && len < (int) sizeof(buf) - 32; ++rank) {
      int best = -1; uint64_t bestv = 0;
      for (int f = 0; f < 16; ++f) if (lh[f] > bestv) { bestv = lh[f]; best = f; }
      if (best < 0) break;
      len += snprintf(buf + len, sizeof(buf) - len, " 0x%x=%llu", best, (unsigned long long) bestv);
      lh[best] = 0;
    }
    printf("%s\n", buf);
  }
  m_stat_native = m_stat_interp = m_stat_hot = m_stat_miss = 0;   // reset the window
}
#endif

#endif // ES40_JIT
