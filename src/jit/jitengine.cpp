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
  OP_LDQ, OP_LDL                 // memory-format loads: Ra = MEM[Rb + disp16]
};

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
    case 0x28: return OP_LDL;   // memory-format loads (Ra = MEM[Rb+disp16])
    case 0x29: return OP_LDQ;
  }
  return OP_NONE;
}

} // namespace

CJitEngine::CJitEngine() : m_recorded(0), m_code_bytes(0), m_rt(nullptr)
{
  flush();                                  // clears valid bits (m_rt still null)
  m_rt = new asmjit::JitRuntime();
#ifdef JIT_VERIFY
  m_v_exec = m_v_fail = 0;
#endif
#ifdef JIT_STATS
  m_stat_native = m_stat_interp = 0;
#endif
}

CJitEngine::~CJitEngine()
{
  delete (asmjit::JitRuntime*) m_rt;
}

CJitEngine::JitBlock* CJitEngine::record(uint64_t virt_pc, uint64_t phys_pc, uint32_t asn, bool asm_global, uint32_t n_instr)
{
  JitBlock& b = m_blocks[index_of(virt_pc)];
  if (b.valid && b.tag == virt_pc && (b.asm_global || b.asn == asn)) {  // re-seen; keep compiled state
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
  b.prefix_len = 0;
  b.compiled = false;
  if (++m_recorded == 50000)
    printf("[JIT] block dispatcher active: 50000 blocks discovered.\n");
  return &b;
}

void CJitEngine::flush()
{
  for (int i = 0; i < kCacheEntries; ++i)
    m_blocks[i].valid = false;
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

void CJitEngine::compile_block(JitBlock* b, const uint8_t* dram, uint64_t dram_size, void* read_helper)
{
  using namespace asmjit;
  b->compiled = true;

  // Only non-PALmode blocks: there RREG(a)==a, so direct state.r[reg] access is
  // correct. PALmode (PC bit 0) can remap R4-7/R20-23 to the shadow bank.
  if (b->tag & 1) return;
  uint64_t phys = b->phys;
  if (b->n_instr == 0 || phys + (uint64_t) b->n_instr * 4 > dram_size) return;
  const uint32_t* words = (const uint32_t*) (dram + phys);   // x86 LE == Alpha LE

  // Stop at the 8 KB page boundary: past it the next instruction's physical
  // address need not be phys+4 (the next virtual page maps elsewhere), so words[]
  // there would be the wrong instructions. (The page-crossing case verify caught.)
  const uint64_t page_end = (phys & ~(uint64_t) 0x1FFF) + 0x2000;
  uint32_t plen = 0;
  while (plen < b->n_instr && plen < 64
         && (phys + (uint64_t) plen * 4) < page_end
         && classify(words[plen]) != OP_NONE)
    plen++;
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
  a.sub(x86::rsp, imm(40));
  a.mov(x86::rsi, x86::rcx);   // cpu
  a.mov(x86::rbx, x86::rdx);   // regs base

  Label done = a.new_label();  // shared bail/return target
  for (uint32_t i = 0; i < plen; ++i) {
    uint32_t ins = words[i];
    SafeOp op = classify(ins);
    int ra = (ins >> 21) & 0x1F;
    int rb = (ins >> 16) & 0x1F;
    int rc = ins & 0x1F;
    bool islit = ((ins >> 12) & 1) != 0;
    uint32_t lit = (ins >> 13) & 0xFF;

    auto reg = [&](int r) { return x86::qword_ptr(x86::rbx, r * 8); };
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
        a.mov(x86::eax, imm(i));                          // fault: bail, i instrs done
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
        else          a.mov(x86::eax, x86::dword_ptr(x86::rbx, ra * 8));
        if (islit)         a.mov(x86::ecx, imm(lit));
        else if (rb == 31) a.xor_(x86::ecx, x86::ecx);
        else               a.mov(x86::ecx, x86::dword_ptr(x86::rbx, rb * 8));
        if (op == OP_ADDL) a.add(x86::eax, x86::ecx);
        else               a.sub(x86::eax, x86::ecx);
        a.movsxd(x86::rax, x86::eax);
        break;
      }
      default: break;
    }

    if (rc != 31) a.mov(reg(rc), x86::rax);
  }
  a.mov(x86::eax, imm(plen));   // no fault: all instructions completed
  a.bind(done);
  a.add(x86::rsp, imm(40));
  a.pop(x86::rsi);
  a.pop(x86::rbx);
  a.ret();

  const size_t csz = code.code_size();
  JitFn fn = nullptr;
  if (((JitRuntime*) m_rt)->add(&fn, &code) != Error::kOk) return;
  b->code = fn;
  b->prefix_len = plen;
  m_code_bytes += csz;   // track for the reclaim threshold (see flush())
}

#ifdef JIT_VERIFY
void CJitEngine::verify_compare(uint64_t blk_virt, const uint64_t* interp, const uint64_t* jit)
{
  m_v_exec++;
  for (int r = 0; r < 31; ++r) {
    if (interp[r] != jit[r]) {
      m_v_fail++;
      printf("[JIT][VERIFY] MISMATCH at block pc=%016llx: r%d interp=%016llx jit=%016llx\n",
             (unsigned long long) blk_virt, r,
             (unsigned long long) interp[r], (unsigned long long) jit[r]);
      break;
    }
  }
  if ((m_v_exec % 500000) == 0)
    printf("[JIT][VERIFY] %llu compiled-block execs, %llu mismatches\n",
           (unsigned long long) m_v_exec, (unsigned long long) m_v_fail);
}
#endif

#ifdef JIT_STATS
void CJitEngine::note_exec(uint32_t native_instr, uint32_t interp_instr)
{
  m_stat_native += native_instr;
  m_stat_interp += interp_instr;
  const uint64_t total = m_stat_native + m_stat_interp;
  if (total >= 100000000)   // report every 100M instructions
  {
    printf("[JIT][STATS] native %.1f%% (%llu of %llu instrs), %llu blocks recorded\n",
           100.0 * (double) m_stat_native / (double) total,
           (unsigned long long) m_stat_native,
           (unsigned long long) total,
           (unsigned long long) m_recorded);
    m_stat_native = m_stat_interp = 0;
  }
}
#endif

#endif // ES40_JIT
