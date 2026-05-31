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
  OP_AND, OP_BIS, OP_XOR, OP_BIC, OP_ORNOT, OP_EQV,
  OP_CMPEQ, OP_CMPLT, OP_CMPLE, OP_CMPULT, OP_CMPULE,
  OP_SLL, OP_SRL, OP_SRA, OP_MULQ,
  OP_LDQ, OP_LDL,                // memory-format loads: Ra = MEM[Rb + disp16]
  OP_STQ, OP_STL,                // memory-format stores: MEM[Rb + disp16] = Ra
  OP_LDA, OP_LDAH,               // load-address: Ra = Rb + disp16 (<<16 for LDAH); pure ALU
  OP_JMP,                        // JMP/JSR/RET (0x1a): Ra = PC+4; PC = Rb & ~3 (computed target)
  OP_CALL_PAL,                   // CALL_PAL (0x00): save R23/exc_addr; PC = pal_base | entry offset
  // Branch-format terminators (contiguous; see is_branch). Conditional on Ra, plus BR/BSR.
  OP_BEQ, OP_BNE, OP_BLT, OP_BLE, OP_BGT, OP_BGE, OP_BLBC, OP_BLBS, OP_BR, OP_BSR
};

static inline bool is_branch(SafeOp op) { return op >= OP_BEQ && op <= OP_BSR; }
static inline bool is_store(SafeOp op)  { return op == OP_STQ || op == OP_STL; }
// A terminator ends the block and writes its own next PC (branches + the computed jump).
static inline bool is_terminator(SafeOp op) { return op == OP_JMP || op == OP_CALL_PAL || is_branch(op); }

// Safe = goto-free, register-only operate-format ops (no trap, memory, or branch).
SafeOp classify(uint32_t ins)
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
        case 0x2d: return OP_CMPEQ;  case 0x4d: return OP_CMPLT;
        case 0x6d: return OP_CMPLE;  case 0x1d: return OP_CMPULT;
        case 0x3d: return OP_CMPULE;
      }
      break;
    case 0x11: // INTL
      switch (func) {
        case 0x00: return OP_AND;   case 0x20: return OP_BIS;
        case 0x40: return OP_XOR;   case 0x08: return OP_BIC;
        case 0x28: return OP_ORNOT; case 0x48: return OP_EQV;
      }
      break;
    case 0x12: // INTS
      switch (func) {
        case 0x39: return OP_SLL; case 0x34: return OP_SRL; case 0x3c: return OP_SRA;
      }
      break;
    case 0x13: // INTM
      if (func == 0x20) return OP_MULQ;
      break;
    case 0x00: {                // CALL_PAL: compile valid standard funcs (priv 0x00-0x3f, unpriv 0x80-0xbf)
      const uint32_t fn = ins & 0x1FFFFFFF;
      if (fn <= 0x3F || (fn >= 0x80 && fn <= 0xBF)) return OP_CALL_PAL;
      break;                    // SRM specials (0x1234xx) / invalid ranges -> interpret
    }
    case 0x08: return OP_LDA;   // load address (Ra = Rb + disp16) -- pure ALU, no memory
    case 0x09: return OP_LDAH;  // load address high (Ra = Rb + (disp16 << 16))
    // NOTE: 0x1a (JMP/JSR/RET) intentionally NOT compiled. It's a terminator (can't lengthen
    // blocks), its targets vary so the single-slot link cache thrashes (no chaining), and the
    // chains are cut by CALL_PAL/MISC anyway -- compiling it measured as a net regression. The
    // OP_JMP codegen/verify paths below stay dormant; revisit once per-dispatch overhead drops.
    case 0x28: return OP_LDL;   // memory-format loads (Ra = MEM[Rb+disp16])
    case 0x29: return OP_LDQ;
    case 0x2c: return OP_STL;   // memory-format stores (MEM[Rb+disp16] = Ra)
    case 0x2d: return OP_STQ;
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

CJitEngine::CJitEngine() : m_recorded(0), m_code_bytes(0), m_rt(nullptr)
{
  flush();                                  // clears valid bits (m_rt still null)
  m_rt = new asmjit::JitRuntime();
#ifdef JIT_VERIFY
  m_v_exec = m_v_fail = 0;
#endif
#ifdef JIT_STATS
  m_stat_native = m_stat_interp = m_stat_hot = m_stat_miss = 0;
  m_stat_compiled = m_stat_plen_sum = 0;
  memset(m_term_op, 0, sizeof(m_term_op));
  memset(m_pal_func, 0, sizeof(m_pal_func));
  m_first_breaker_logged = false;
#endif
}

CJitEngine::~CJitEngine()
{
  delete (asmjit::JitRuntime*) m_rt;
}

CJitEngine::JitBlock* CJitEngine::record(uint64_t virt_pc, uint64_t phys_pc, uint32_t asn, bool asm_global, uint32_t n_instr)
{
  JitBlock& b = m_blocks[index_of(virt_pc)];
  // Re-seen only if the live physical still matches: virtual+ASN keying can't see a
  // page remap, so a changed phys means the recorded bytes are stale -- fall through
  // and re-record (clears code/compiled below, forcing a recompile from the new phys).
  if (b.valid && b.tag == virt_pc && (b.asm_global || b.asn == asn) && b.phys == phys_pc) {
    b.n_instr = n_instr;
    return &b;
  }
  b.tag = virt_pc;
  b.phys = phys_pc;
  b.asn = asn;
  b.asm_global = asm_global;
  b.n_instr = n_instr;
  b.valid = true;
  b.code = nullptr;
  b.jit_body = nullptr;   // not compiled yet -> cached links to us must miss until compile
  b.link = nullptr;       // no cached successor yet
  b.prefix_len = 0;
  b.compiled = false;
  if (++m_recorded == 50000)
    printf("[JIT] block dispatcher active: 50000 blocks discovered.\n");
  return &b;
}

void CJitEngine::flush()
{
  for (int i = 0; i < kCacheEntries; ++i) {
    m_blocks[i].valid = false;
    m_blocks[i].jit_body = nullptr;   // break stale cached links: the epilogue gates jumps on jit_body
  }
  // flush() runs on every TB invalidation (frequent); tearing down the asmjit
  // runtime each time was the dominant cost. Code from invalidated blocks is
  // unreachable, so only reclaim once it has grown past the cap. (delete+new, not
  // reset()-and-reuse, which corrupted the JitAllocator block tree.)
  if (m_rt && m_code_bytes >= kReclaimBytes)
  {
    delete (asmjit::JitRuntime*) m_rt;
    m_rt = new asmjit::JitRuntime();
    m_code_bytes = 0;
  }
}

void CJitEngine::compile_block(JitBlock* b, const uint8_t* dram, uint64_t dram_size, void* read_helper, void* write_helper, void* opcdec_helper)
{
  using namespace asmjit;
  b->compiled = true;

  uint64_t phys = b->phys;
  if (b->n_instr == 0 || phys + (uint64_t) b->n_instr * 4 > dram_size) return;
  const uint32_t* words = (const uint32_t*) (dram + phys);   // x86 LE == Alpha LE

  // PALmode blocks (PC bit 0) can remap R4-7/R20-23 to the shadow bank (see RREG), so direct
  // state.r[reg] access isn't valid there. 
  const bool pal_block = (b->tag & 1) != 0;   // PALmode (PC bit 0); reg() applies the shadow remap

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
    SafeOp sop = classify(words[plen]);
    if (sop == OP_NONE) {               // uncompilable op ends the straight-line prefix
#ifdef JIT_STATS
      const uint32_t bop = words[plen] >> 26;
      m_term_op[bop]++;                 // tally what cut this block (the coverage gap to chase)
      if (bop == 0x00)                  // CALL_PAL: also tally the function code (low 8 bits)
        m_pal_func[words[plen] & 0xFF]++;
      // Punch list: one-shot print of the first ACTIONABLE breaker
      if (!m_first_breaker_logged && bop != 0x1a && bop != 0x00) {
        m_first_breaker_logged = true;
        printf("[JIT][PUNCH] first unhandled breaker: %s(0x%02x) ins=%08x at pc=%016llx%s\n",
               opcode_name(bop), bop, words[plen],
               (unsigned long long) ((b->tag & ~(uint64_t) 1) + (uint64_t) plen * 4),
               pal_block ? "  [PALmode]" : "");
      }
#endif
      break;
    }
    plen++;
    if (is_terminator(sop)) {           // branch or computed jump ends the block
      terminator_branch = true;
      if (sop == OP_JMP) terminator_jmp = true;
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
    SafeOp op = classify(ins);
    int ra = (ins >> 21) & 0x1F;
    int rb = (ins >> 16) & 0x1F;
    int rc = ins & 0x1F;
    bool islit = ((ins >> 12) & 1) != 0;
    uint32_t lit = (ins >> 13) & 0xFF;

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
    if (op == OP_LDQ || op == OP_LDL) {
      if (ra == 31) continue;            // LDx R31 is a NOP (interpreter skips the read)
      const int disp = (int) (int16_t) (ins & 0xFFFF);
      const int size_bits = (op == OP_LDQ) ? 64 : 32;
      const int amask = (size_bits / 8) - 1;
      if (rb == 31)  a.mov(x86::rdx, imm(disp));         // va -> RDX
      else        {  a.mov(x86::rdx, reg(rb)); if (disp) a.add(x86::rdx, imm(disp)); }

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
        if (op == OP_LDQ) a.mov(x86::rax, x86::qword_ptr(x86::rsp, 32));
        else              a.movsxd(x86::rax, x86::dword_ptr(x86::rsp, 32));
        a.mov(reg(ra), x86::rax);
      };

#ifdef JIT_VERIFY
      emit_helper();
#else
      // Inline fast path: aligned + data_page_cache[0] hit + DRAM. Falls to the helper
      // on misalign / cache miss / MMIO. Mirrors jit_read's data-cache path. RDX = va
      // (preserved across the checks); RAX/R10 scratch.
      Label slow  = a.new_label();
      Label ldone = a.new_label();
      a.test(x86::dl, imm(amask));                                      a.jnz(slow);
      a.mov(x86::r10, x86::rdx);
      a.and_(x86::r10, imm(-0x2000));                                   // r10 = va & ~0x1FFF
      a.cmp(x86::byte_ptr(x86::rsi, m_off.dpc_valid), imm(0));          a.je(slow);
      a.cmp(x86::qword_ptr(x86::rsi, m_off.dpc_virt_page), x86::r10);   a.jne(slow);
      a.mov(x86::eax, x86::dword_ptr(x86::rsi, m_off.state_cm));
      a.cmp(x86::dword_ptr(x86::rsi, m_off.dpc_cm), x86::eax);          a.jne(slow);
      a.mov(x86::eax, x86::dword_ptr(x86::rsi, m_off.state_asn0));
      a.cmp(x86::dword_ptr(x86::rsi, m_off.dpc_asn), x86::eax);         a.jne(slow);
      a.mov(x86::rax, x86::qword_ptr(x86::rsi, m_off.dpc_phys_base));
      a.mov(x86::r10, x86::rdx);
      a.and_(x86::r10, imm(0x1FFF));
      a.or_(x86::rax, x86::r10);                                        // rax = phys
      a.cmp(x86::rax, x86::qword_ptr(x86::rsi, m_off.dram_size));       a.jae(slow);
      a.mov(x86::r10, x86::qword_ptr(x86::rsi, m_off.dram_ptr));
      if (op == OP_LDQ) a.mov(x86::rax, x86::qword_ptr(x86::r10, x86::rax));
      else              a.movsxd(x86::rax, x86::dword_ptr(x86::r10, x86::rax));
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
    if (op == OP_STL || op == OP_STQ) {
      const int disp = (int) (int16_t) (ins & 0xFFFF);
      const int size_bits = (op == OP_STQ) ? 64 : 32;
      if (rb == 31)  a.mov(x86::rdx, imm(disp));                       // va -> RDX
      else        {  a.mov(x86::rdx, reg(rb)); if (disp) a.add(x86::rdx, imm(disp)); }
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

    // Computed jump JMP/JSR/RET (0x1a): Ra = PC+4 (return address); PC = Rb & ~3. A
    // terminator like a branch, but the target is a register -- left in R10 so the epilogue's
    // cached link validates it (succ->tag == target), chaining calls/returns/dispatch.
    if (op == OP_JMP) {
      const uint64_t ret = b->tag + 4 * (uint64_t) (i + 1);
      if (rb == 31) a.xor_(x86::r10d, x86::r10d);
      else          a.mov(x86::r10, reg(rb));
      a.and_(x86::r10, imm(~(uint64_t) 3));                       // target = Rb & ~3 (clear low 2)
      if (ra != 31) { a.mov(x86::rax, imm(ret & ~(uint64_t) 3)); a.mov(reg(ra), x86::rax); }  // return addr = PC & ~3 (DO_JMP)
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

      case OP_S4ADDQ: op1_rax(); a.shl(x86::rax, imm(2)); op2_rcx(); a.add(x86::rax, x86::rcx); break;
      case OP_S8ADDQ: op1_rax(); a.shl(x86::rax, imm(3)); op2_rcx(); a.add(x86::rax, x86::rcx); break;
      case OP_S4SUBQ: op1_rax(); a.shl(x86::rax, imm(2)); op2_rcx(); a.sub(x86::rax, x86::rcx); break;
      case OP_S8SUBQ: op1_rax(); a.shl(x86::rax, imm(3)); op2_rcx(); a.sub(x86::rax, x86::rcx); break;

      case OP_SLL: op1_rax(); op2_rcx(); a.shl(x86::rax, x86::cl); break;
      case OP_SRL: op1_rax(); op2_rcx(); a.shr(x86::rax, x86::cl); break;
      case OP_SRA: op1_rax(); op2_rcx(); a.sar(x86::rax, x86::cl); break;

      case OP_CMPEQ:  op1_rax(); op2_rcx(); a.cmp(x86::rax, x86::rcx); a.sete(x86::al);  a.movzx(x86::eax, x86::al); break;
      case OP_CMPLT:  op1_rax(); op2_rcx(); a.cmp(x86::rax, x86::rcx); a.setl(x86::al);  a.movzx(x86::eax, x86::al); break;
      case OP_CMPLE:  op1_rax(); op2_rcx(); a.cmp(x86::rax, x86::rcx); a.setle(x86::al); a.movzx(x86::eax, x86::al); break;
      case OP_CMPULT: op1_rax(); op2_rcx(); a.cmp(x86::rax, x86::rcx); a.setb(x86::al);  a.movzx(x86::eax, x86::al); break;
      case OP_CMPULE: op1_rax(); op2_rcx(); a.cmp(x86::rax, x86::rcx); a.setbe(x86::al); a.movzx(x86::eax, x86::al); break;

      case OP_ADDL: case OP_SUBL:  // 32-bit, result sign-extended to 64
      {
        if (ra == 31) a.xor_(x86::eax, x86::eax);
        else          a.mov(x86::eax, reg32(ra));   // shadow-remapped (was a raw rbx read)
        if (islit)         a.mov(x86::ecx, imm(lit));
        else if (rb == 31) a.xor_(x86::ecx, x86::ecx);
        else               a.mov(x86::ecx, reg32(rb));
        if (op == OP_ADDL) a.add(x86::eax, x86::ecx);
        else               a.sub(x86::eax, x86::ecx);
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
  const uint32_t off_body = (uint32_t) ((char*) &b->jit_body - (char*) b);
  const uint32_t off_tag  = (uint32_t) ((char*) &b->tag      - (char*) b);
  // Cached direct link: tail straight into our cached successor's body if it's still live
  // -- compiled (jit_body != 0, cleared on flush/recompile) AND still maps this exit's PC
  // (tag == R10). Otherwise record a patch request (m_link_from = this block) so the
  // dispatcher fills b->link the next time it runs the successor, and fall back to it.
  // R10 = next PC; clobbers RAX/RCX/RDX.
  auto emit_chain = [&](Label& lbl) {
    Label miss = a.new_label();
    a.mov(x86::rax, imm((uint64_t) &b->link));
    a.mov(x86::rax, x86::qword_ptr(x86::rax));                 // succ = b->link
    a.test(x86::rax, x86::rax);                            a.jz(miss);
    a.mov(x86::rcx, x86::qword_ptr(x86::rax, off_body));       // succ->jit_body
    a.test(x86::rcx, x86::rcx);                            a.jz(miss);
    a.mov(x86::rdx, x86::qword_ptr(x86::rax, off_tag));        // succ->tag
    a.cmp(x86::rdx, x86::r10);                             a.jne(miss);   // not this exit's target
    a.jmp(x86::rcx);                                           // HIT: tail in (shared frame)
    a.bind(miss);
    a.mov(x86::rax, imm((uint64_t) b));
    a.mov(x86::qword_ptr(x86::rsi, m_off.link_from), x86::rax);   // request b->link patch
    // fall through to lbl (return to dispatcher)
  };
#endif
  if (terminator_jmp) {
    // Computed jump: the JMP codegen already wrote state.pc to the (register) target. Don't
    // chain it -- targets vary (returns/dispatch), so the single-slot cache thrashes, costing
    // more than the dispatcher round-trip it would save (measured regression). Just count the
    // block and return; the dispatcher resolves the target (and may run a fresh chain there).
    a.add(x86::r14d, imm(plen));
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

  const double chain = m_stat_hot ? (double) m_stat_native / (double) m_stat_hot : 0.0;
  const double avgpl = m_stat_compiled ? (double) m_stat_plen_sum / (double) m_stat_compiled : 0.0;
  printf("[JIT][STATS] native %.1f%% (%llu/%llu) | chain avg %.1f instr over %llu dispatches | interp %llu blks\n",
         100.0 * (double) m_stat_native / (double) total,
         (unsigned long long) m_stat_native, (unsigned long long) total,
         chain, (unsigned long long) m_stat_hot, (unsigned long long) m_stat_miss);
  printf("[JIT][STATS] %llu recorded, %llu compiled (avg %.1f instr) | block-breakers:",
         (unsigned long long) m_recorded, (unsigned long long) m_stat_compiled, avgpl);
  uint64_t hist[64];
  memcpy(hist, m_term_op, sizeof(hist));
  for (int rank = 0; rank < 8; ++rank) {
    int best = -1; uint64_t bestv = 0;
    for (int op = 0; op < 64; ++op) if (hist[op] > bestv) { bestv = hist[op]; best = op; }
    if (best < 0) break;
    printf(" %s(0x%02x)=%llu", opcode_name(best), best, (unsigned long long) bestv);
    hist[best] = 0;
  }
  printf("\n");
  if (m_term_op[0]) {   // CALL_PAL cut blocks -- show which function codes dominate (chain targets)
    printf("[JIT][STATS]   CALL_PAL by func:");
    uint64_t ph[256];
    memcpy(ph, m_pal_func, sizeof(ph));
    for (int rank = 0; rank < 8; ++rank) {
      int best = -1; uint64_t bestv = 0;
      for (int f = 0; f < 256; ++f) if (ph[f] > bestv) { bestv = ph[f]; best = f; }
      if (best < 0) break;
      printf(" 0x%02x=%llu", best, (unsigned long long) bestv);
      ph[best] = 0;
    }
    printf("\n");
  }
  m_stat_native = m_stat_interp = m_stat_hot = m_stat_miss = 0;   // reset the window
}
#endif

#endif // ES40_JIT
