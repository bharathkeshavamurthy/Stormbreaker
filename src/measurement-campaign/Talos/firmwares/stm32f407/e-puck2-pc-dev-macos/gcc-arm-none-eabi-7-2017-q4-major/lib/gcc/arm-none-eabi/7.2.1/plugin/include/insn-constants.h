/* Generated automatically by the program `genconstants'
   from the machine description file `md'.  */

#ifndef GCC_INSN_CONSTANTS_H
#define GCC_INSN_CONSTANTS_H

#define CMP_CMN 2
#define CMP_CMP 0
#define DOM_CC_NX_OR_Y 1
#define DOM_CC_X_OR_Y 2
#define NUM_OF_COND_CMP 4
#define CC_REGNUM 100
#define WCGR3 99
#define SP_REGNUM 13
#define R1_REGNUM 1
#define PC_REGNUM 15
#define WCGR2 98
#define WCGR0 96
#define VFPCC_REGNUM 101
#define R0_REGNUM 0
#define CMN_CMP 1
#define LR_REGNUM 14
#define WCGR1 97
#define DOM_CC_X_AND_Y 0
#define CMN_CMN 3
#define IP_REGNUM 12
#define LAST_ARM_REGNUM 15

enum unspec {
  UNSPEC_PUSH_MULT = 0,
  UNSPEC_PIC_SYM = 1,
  UNSPEC_PIC_BASE = 2,
  UNSPEC_PRLG_STK = 3,
  UNSPEC_REGISTER_USE = 4,
  UNSPEC_CHECK_ARCH = 5,
  UNSPEC_WSHUFH = 6,
  UNSPEC_WACC = 7,
  UNSPEC_TMOVMSK = 8,
  UNSPEC_WSAD = 9,
  UNSPEC_WSADZ = 10,
  UNSPEC_WMACS = 11,
  UNSPEC_WMACU = 12,
  UNSPEC_WMACSZ = 13,
  UNSPEC_WMACUZ = 14,
  UNSPEC_CLRDI = 15,
  UNSPEC_WALIGNI = 16,
  UNSPEC_TLS = 17,
  UNSPEC_PIC_LABEL = 18,
  UNSPEC_PIC_OFFSET = 19,
  UNSPEC_GOTSYM_OFF = 20,
  UNSPEC_THUMB1_CASESI = 21,
  UNSPEC_RBIT = 22,
  UNSPEC_SYMBOL_OFFSET = 23,
  UNSPEC_MEMORY_BARRIER = 24,
  UNSPEC_UNALIGNED_LOAD = 25,
  UNSPEC_UNALIGNED_STORE = 26,
  UNSPEC_PIC_UNIFIED = 27,
  UNSPEC_LL = 28,
  UNSPEC_VRINTZ = 29,
  UNSPEC_VRINTP = 30,
  UNSPEC_VRINTM = 31,
  UNSPEC_VRINTR = 32,
  UNSPEC_VRINTX = 33,
  UNSPEC_VRINTA = 34,
  UNSPEC_PROBE_STACK = 35,
  UNSPEC_NONSECURE_MEM = 36,
  UNSPEC_WADDC = 37,
  UNSPEC_WABS = 38,
  UNSPEC_WQMULWMR = 39,
  UNSPEC_WQMULMR = 40,
  UNSPEC_WQMULWM = 41,
  UNSPEC_WQMULM = 42,
  UNSPEC_WQMIAxyn = 43,
  UNSPEC_WQMIAxy = 44,
  UNSPEC_TANDC = 45,
  UNSPEC_TORC = 46,
  UNSPEC_TORVSC = 47,
  UNSPEC_TEXTRC = 48,
  UNSPEC_ASHIFT_SIGNED = 49,
  UNSPEC_ASHIFT_UNSIGNED = 50,
  UNSPEC_CRC32B = 51,
  UNSPEC_CRC32H = 52,
  UNSPEC_CRC32W = 53,
  UNSPEC_CRC32CB = 54,
  UNSPEC_CRC32CH = 55,
  UNSPEC_CRC32CW = 56,
  UNSPEC_AESD = 57,
  UNSPEC_AESE = 58,
  UNSPEC_AESIMC = 59,
  UNSPEC_AESMC = 60,
  UNSPEC_SHA1C = 61,
  UNSPEC_SHA1M = 62,
  UNSPEC_SHA1P = 63,
  UNSPEC_SHA1H = 64,
  UNSPEC_SHA1SU0 = 65,
  UNSPEC_SHA1SU1 = 66,
  UNSPEC_SHA256H = 67,
  UNSPEC_SHA256H2 = 68,
  UNSPEC_SHA256SU0 = 69,
  UNSPEC_SHA256SU1 = 70,
  UNSPEC_VMULLP64 = 71,
  UNSPEC_LOAD_COUNT = 72,
  UNSPEC_VABD_F = 73,
  UNSPEC_VABD_S = 74,
  UNSPEC_VABD_U = 75,
  UNSPEC_VABDL_S = 76,
  UNSPEC_VABDL_U = 77,
  UNSPEC_VADD = 78,
  UNSPEC_VADDHN = 79,
  UNSPEC_VRADDHN = 80,
  UNSPEC_VADDL_S = 81,
  UNSPEC_VADDL_U = 82,
  UNSPEC_VADDW_S = 83,
  UNSPEC_VADDW_U = 84,
  UNSPEC_VBSL = 85,
  UNSPEC_VCAGE = 86,
  UNSPEC_VCAGT = 87,
  UNSPEC_VCALE = 88,
  UNSPEC_VCALT = 89,
  UNSPEC_VCEQ = 90,
  UNSPEC_VCGE = 91,
  UNSPEC_VCGEU = 92,
  UNSPEC_VCGT = 93,
  UNSPEC_VCGTU = 94,
  UNSPEC_VCLS = 95,
  UNSPEC_VCONCAT = 96,
  UNSPEC_VCVT = 97,
  UNSPEC_VCVT_S = 98,
  UNSPEC_VCVT_U = 99,
  UNSPEC_VCVT_S_N = 100,
  UNSPEC_VCVT_U_N = 101,
  UNSPEC_VCVT_HF_S_N = 102,
  UNSPEC_VCVT_HF_U_N = 103,
  UNSPEC_VCVT_SI_S_N = 104,
  UNSPEC_VCVT_SI_U_N = 105,
  UNSPEC_VCVTH_S = 106,
  UNSPEC_VCVTH_U = 107,
  UNSPEC_VCVTA_S = 108,
  UNSPEC_VCVTA_U = 109,
  UNSPEC_VCVTM_S = 110,
  UNSPEC_VCVTM_U = 111,
  UNSPEC_VCVTN_S = 112,
  UNSPEC_VCVTN_U = 113,
  UNSPEC_VCVTP_S = 114,
  UNSPEC_VCVTP_U = 115,
  UNSPEC_VEXT = 116,
  UNSPEC_VHADD_S = 117,
  UNSPEC_VHADD_U = 118,
  UNSPEC_VRHADD_S = 119,
  UNSPEC_VRHADD_U = 120,
  UNSPEC_VHSUB_S = 121,
  UNSPEC_VHSUB_U = 122,
  UNSPEC_VLD1 = 123,
  UNSPEC_VLD1_LANE = 124,
  UNSPEC_VLD2 = 125,
  UNSPEC_VLD2_DUP = 126,
  UNSPEC_VLD2_LANE = 127,
  UNSPEC_VLD3 = 128,
  UNSPEC_VLD3A = 129,
  UNSPEC_VLD3B = 130,
  UNSPEC_VLD3_DUP = 131,
  UNSPEC_VLD3_LANE = 132,
  UNSPEC_VLD4 = 133,
  UNSPEC_VLD4A = 134,
  UNSPEC_VLD4B = 135,
  UNSPEC_VLD4_DUP = 136,
  UNSPEC_VLD4_LANE = 137,
  UNSPEC_VMAX = 138,
  UNSPEC_VMAX_U = 139,
  UNSPEC_VMAXNM = 140,
  UNSPEC_VMIN = 141,
  UNSPEC_VMIN_U = 142,
  UNSPEC_VMINNM = 143,
  UNSPEC_VMLA = 144,
  UNSPEC_VMLA_LANE = 145,
  UNSPEC_VMLAL_S = 146,
  UNSPEC_VMLAL_U = 147,
  UNSPEC_VMLAL_S_LANE = 148,
  UNSPEC_VMLAL_U_LANE = 149,
  UNSPEC_VMLS = 150,
  UNSPEC_VMLS_LANE = 151,
  UNSPEC_VMLSL_S = 152,
  UNSPEC_VMLSL_U = 153,
  UNSPEC_VMLSL_S_LANE = 154,
  UNSPEC_VMLSL_U_LANE = 155,
  UNSPEC_VMLSL_LANE = 156,
  UNSPEC_VFMA_LANE = 157,
  UNSPEC_VFMS_LANE = 158,
  UNSPEC_VMOVL_S = 159,
  UNSPEC_VMOVL_U = 160,
  UNSPEC_VMOVN = 161,
  UNSPEC_VMUL = 162,
  UNSPEC_VMULL_P = 163,
  UNSPEC_VMULL_S = 164,
  UNSPEC_VMULL_U = 165,
  UNSPEC_VMUL_LANE = 166,
  UNSPEC_VMULL_S_LANE = 167,
  UNSPEC_VMULL_U_LANE = 168,
  UNSPEC_VPADAL_S = 169,
  UNSPEC_VPADAL_U = 170,
  UNSPEC_VPADD = 171,
  UNSPEC_VPADDL_S = 172,
  UNSPEC_VPADDL_U = 173,
  UNSPEC_VPMAX = 174,
  UNSPEC_VPMAX_U = 175,
  UNSPEC_VPMIN = 176,
  UNSPEC_VPMIN_U = 177,
  UNSPEC_VPSMAX = 178,
  UNSPEC_VPSMIN = 179,
  UNSPEC_VPUMAX = 180,
  UNSPEC_VPUMIN = 181,
  UNSPEC_VQABS = 182,
  UNSPEC_VQADD_S = 183,
  UNSPEC_VQADD_U = 184,
  UNSPEC_VQDMLAL = 185,
  UNSPEC_VQDMLAL_LANE = 186,
  UNSPEC_VQDMLSL = 187,
  UNSPEC_VQDMLSL_LANE = 188,
  UNSPEC_VQDMULH = 189,
  UNSPEC_VQDMULH_LANE = 190,
  UNSPEC_VQRDMULH = 191,
  UNSPEC_VQRDMULH_LANE = 192,
  UNSPEC_VQDMULL = 193,
  UNSPEC_VQDMULL_LANE = 194,
  UNSPEC_VQMOVN_S = 195,
  UNSPEC_VQMOVN_U = 196,
  UNSPEC_VQMOVUN = 197,
  UNSPEC_VQNEG = 198,
  UNSPEC_VQSHL_S = 199,
  UNSPEC_VQSHL_U = 200,
  UNSPEC_VQRSHL_S = 201,
  UNSPEC_VQRSHL_U = 202,
  UNSPEC_VQSHL_S_N = 203,
  UNSPEC_VQSHL_U_N = 204,
  UNSPEC_VQSHLU_N = 205,
  UNSPEC_VQSHRN_S_N = 206,
  UNSPEC_VQSHRN_U_N = 207,
  UNSPEC_VQRSHRN_S_N = 208,
  UNSPEC_VQRSHRN_U_N = 209,
  UNSPEC_VQSHRUN_N = 210,
  UNSPEC_VQRSHRUN_N = 211,
  UNSPEC_VQSUB_S = 212,
  UNSPEC_VQSUB_U = 213,
  UNSPEC_VRECPE = 214,
  UNSPEC_VRECPS = 215,
  UNSPEC_VREV16 = 216,
  UNSPEC_VREV32 = 217,
  UNSPEC_VREV64 = 218,
  UNSPEC_VRSQRTE = 219,
  UNSPEC_VRSQRTS = 220,
  UNSPEC_VSHL_S = 221,
  UNSPEC_VSHL_U = 222,
  UNSPEC_VRSHL_S = 223,
  UNSPEC_VRSHL_U = 224,
  UNSPEC_VSHLL_S_N = 225,
  UNSPEC_VSHLL_U_N = 226,
  UNSPEC_VSHL_N = 227,
  UNSPEC_VSHR_S_N = 228,
  UNSPEC_VSHR_U_N = 229,
  UNSPEC_VRSHR_S_N = 230,
  UNSPEC_VRSHR_U_N = 231,
  UNSPEC_VSHRN_N = 232,
  UNSPEC_VRSHRN_N = 233,
  UNSPEC_VSLI = 234,
  UNSPEC_VSRA_S_N = 235,
  UNSPEC_VSRA_U_N = 236,
  UNSPEC_VRSRA_S_N = 237,
  UNSPEC_VRSRA_U_N = 238,
  UNSPEC_VSRI = 239,
  UNSPEC_VST1 = 240,
  UNSPEC_VST1_LANE = 241,
  UNSPEC_VST2 = 242,
  UNSPEC_VST2_LANE = 243,
  UNSPEC_VST3 = 244,
  UNSPEC_VST3A = 245,
  UNSPEC_VST3B = 246,
  UNSPEC_VST3_LANE = 247,
  UNSPEC_VST4 = 248,
  UNSPEC_VST4A = 249,
  UNSPEC_VST4B = 250,
  UNSPEC_VST4_LANE = 251,
  UNSPEC_VSTRUCTDUMMY = 252,
  UNSPEC_VSUB = 253,
  UNSPEC_VSUBHN = 254,
  UNSPEC_VRSUBHN = 255,
  UNSPEC_VSUBL_S = 256,
  UNSPEC_VSUBL_U = 257,
  UNSPEC_VSUBW_S = 258,
  UNSPEC_VSUBW_U = 259,
  UNSPEC_VTBL = 260,
  UNSPEC_VTBX = 261,
  UNSPEC_VTRN1 = 262,
  UNSPEC_VTRN2 = 263,
  UNSPEC_VTST = 264,
  UNSPEC_VUZP1 = 265,
  UNSPEC_VUZP2 = 266,
  UNSPEC_VZIP1 = 267,
  UNSPEC_VZIP2 = 268,
  UNSPEC_MISALIGNED_ACCESS = 269,
  UNSPEC_VCLE = 270,
  UNSPEC_VCLT = 271,
  UNSPEC_NVRINTZ = 272,
  UNSPEC_NVRINTP = 273,
  UNSPEC_NVRINTM = 274,
  UNSPEC_NVRINTX = 275,
  UNSPEC_NVRINTA = 276,
  UNSPEC_NVRINTN = 277,
  UNSPEC_VQRDMLAH = 278,
  UNSPEC_VQRDMLSH = 279,
  UNSPEC_VRND = 280,
  UNSPEC_VRNDA = 281,
  UNSPEC_VRNDI = 282,
  UNSPEC_VRNDM = 283,
  UNSPEC_VRNDN = 284,
  UNSPEC_VRNDP = 285,
  UNSPEC_VRNDX = 286
};
#define NUM_UNSPEC_VALUES 287
extern const char *const unspec_strings[];

enum unspecv {
  VUNSPEC_BLOCKAGE = 0,
  VUNSPEC_EPILOGUE = 1,
  VUNSPEC_THUMB1_INTERWORK = 2,
  VUNSPEC_ALIGN = 3,
  VUNSPEC_POOL_END = 4,
  VUNSPEC_POOL_1 = 5,
  VUNSPEC_POOL_2 = 6,
  VUNSPEC_POOL_4 = 7,
  VUNSPEC_POOL_8 = 8,
  VUNSPEC_POOL_16 = 9,
  VUNSPEC_TMRC = 10,
  VUNSPEC_TMCR = 11,
  VUNSPEC_ALIGN8 = 12,
  VUNSPEC_WCMP_EQ = 13,
  VUNSPEC_WCMP_GTU = 14,
  VUNSPEC_WCMP_GT = 15,
  VUNSPEC_EH_RETURN = 16,
  VUNSPEC_ATOMIC_CAS = 17,
  VUNSPEC_ATOMIC_XCHG = 18,
  VUNSPEC_ATOMIC_OP = 19,
  VUNSPEC_LL = 20,
  VUNSPEC_LDRD_ATOMIC = 21,
  VUNSPEC_SC = 22,
  VUNSPEC_LAX = 23,
  VUNSPEC_SLX = 24,
  VUNSPEC_LDA = 25,
  VUNSPEC_STL = 26,
  VUNSPEC_GET_FPSCR = 27,
  VUNSPEC_SET_FPSCR = 28,
  VUNSPEC_PROBE_STACK_RANGE = 29,
  VUNSPEC_CDP = 30,
  VUNSPEC_CDP2 = 31,
  VUNSPEC_LDC = 32,
  VUNSPEC_LDC2 = 33,
  VUNSPEC_LDCL = 34,
  VUNSPEC_LDC2L = 35,
  VUNSPEC_STC = 36,
  VUNSPEC_STC2 = 37,
  VUNSPEC_STCL = 38,
  VUNSPEC_STC2L = 39,
  VUNSPEC_MCR = 40,
  VUNSPEC_MCR2 = 41,
  VUNSPEC_MRC = 42,
  VUNSPEC_MRC2 = 43,
  VUNSPEC_MCRR = 44,
  VUNSPEC_MCRR2 = 45,
  VUNSPEC_MRRC = 46,
  VUNSPEC_MRRC2 = 47
};
#define NUM_UNSPECV_VALUES 48
extern const char *const unspecv_strings[];

#endif /* GCC_INSN_CONSTANTS_H */
