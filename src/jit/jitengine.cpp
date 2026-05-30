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
  OP_SLL, OP_SRL, OP_SRA, OP_MULQ
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

void CJitEngine::compile_block(JitBlock* b, const uint8_t* dram, uint64_t dram_size)
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

  // Emit  void fn(uint64_t* regs).  regs arrives in RCX (Win64); move it to R8 so
  // RCX/CL is free for variable shifts. RAX = op1/result, RCX = operand2.
  CodeHolder code;
  if (code.init(((JitRuntime*) m_rt)->environment()) != Error::kOk) return;
  x86::Assembler a(&code);
  a.mov(x86::r8, x86::rcx);

  for (uint32_t i = 0; i < plen; ++i) {
    uint32_t ins = words[i];
    SafeOp op = classify(ins);
    int ra = (ins >> 21) & 0x1F;
    int rb = (ins >> 16) & 0x1F;
    int rc = ins & 0x1F;
    bool islit = ((ins >> 12) & 1) != 0;
    uint32_t lit = (ins >> 13) & 0xFF;

    auto reg = [&](int r) { return x86::qword_ptr(x86::r8, r * 8); };
    auto op1_rax = [&]() {
      if (ra == 31) a.xor_(x86::eax, x86::eax);
      else          a.mov(x86::rax, reg(ra));
    };
    auto op2_rcx = [&]() {                  // operand2 (literal, or r[Rb] with r31=0)
      if (islit)         a.mov(x86::rcx, imm(lit));
      else if (rb == 31) a.xor_(x86::ecx, x86::ecx);
      else               a.mov(x86::rcx, reg(rb));
    };
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
        else          a.mov(x86::eax, x86::dword_ptr(x86::r8, ra * 8));
        if (islit)         a.mov(x86::ecx, imm(lit));
        else if (rb == 31) a.xor_(x86::ecx, x86::ecx);
        else               a.mov(x86::ecx, x86::dword_ptr(x86::r8, rb * 8));
        if (op == OP_ADDL) a.add(x86::eax, x86::ecx);
        else               a.sub(x86::eax, x86::ecx);
        a.movsxd(x86::rax, x86::eax);
        break;
      }
      default: break;
    }

    if (rc != 31) a.mov(reg(rc), x86::rax);
  }
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

#endif // ES40_JIT
