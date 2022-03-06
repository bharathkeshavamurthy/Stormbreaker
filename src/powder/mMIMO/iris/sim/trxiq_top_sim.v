// Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2018.3 (lin64) Build 2405991 Thu Dec  6 23:36:41 MST 2018
// Date        : Wed Nov 18 10:56:33 2020
// Host        : bender.ad.sklk.us running 64-bit Ubuntu 16.04.6 LTS
// Command     : write_verilog -force -mode funcsim trxiq_top_sim.v
// Design      : trxiq_top
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7z030sbg485-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* NotValidForBitStream *)
module trxiq_top
   (SYS_clk,
    SYS_rst,
    tick,
    SYS_config_write,
    SYS_config_data,
    SYS_test_data_a_tx,
    SYS_test_data_b_tx,
    SYS_prbs_ctrl,
    SYS_test_data_a_rx,
    SYS_test_data_b_rx,
    SYS_data_clk_counter,
    SYS_prbs_stat,
    SYS_prbs_e,
    DATA_clk,
    DATA_rst,
    RX_data_a,
    RX_data_b,
    TX_data_a,
    TX_data_b,
    LMS_DIQ1_IQSEL,
    LMS_DIQ1_MCLK,
    LMS_DIQ1_FCLK,
    LMS_DIQ1_TXNRX,
    LMS_DIQ1_D,
    LMS_DIQ2_IQSEL,
    LMS_DIQ2_MCLK,
    LMS_DIQ2_FCLK,
    LMS_DIQ2_TXNRX,
    LMS_DIQ2_D);
  input SYS_clk;
  input SYS_rst;
  input tick;
  input SYS_config_write;
  input [31:0]SYS_config_data;
  input [31:0]SYS_test_data_a_tx;
  input [31:0]SYS_test_data_b_tx;
  input [15:0]SYS_prbs_ctrl;
  output [31:0]SYS_test_data_a_rx;
  output [31:0]SYS_test_data_b_rx;
  output [31:0]SYS_data_clk_counter;
  output [31:0]SYS_prbs_stat;
  output [63:0]SYS_prbs_e;
  output DATA_clk;
  output DATA_rst;
  output [31:0]RX_data_a;
  output [31:0]RX_data_b;
  input [31:0]TX_data_a;
  input [31:0]TX_data_b;
  output LMS_DIQ1_IQSEL;
  input LMS_DIQ1_MCLK;
  output LMS_DIQ1_FCLK;
  output LMS_DIQ1_TXNRX;
  output [11:0]LMS_DIQ1_D;
  input LMS_DIQ2_IQSEL;
  input LMS_DIQ2_MCLK;
  output LMS_DIQ2_FCLK;
  output LMS_DIQ2_TXNRX;
  input [11:0]LMS_DIQ2_D;

  wire \<const0> ;
  wire \<const1> ;
  wire DATA_clk;
  wire DATA_rst;
  wire EXT_rst;
  wire EXT_rst_i_1_n_0;
  wire [11:0]LMS_DIQ1_D;
  wire LMS_DIQ1_FCLK;
  wire LMS_DIQ1_IQSEL;
  wire LMS_DIQ1_MCLK;
  wire [11:0]LMS_DIQ2_D;
  wire LMS_DIQ2_IQSEL;
  wire LMS_DIQ2_MCLK;
  wire PRBS_test_signal;
  wire [31:0]RX_data_a;
  wire \RX_data_a[0]_i_1_n_0 ;
  wire \RX_data_a[10]_i_1_n_0 ;
  wire \RX_data_a[11]_i_1_n_0 ;
  wire \RX_data_a[12]_i_1_n_0 ;
  wire \RX_data_a[13]_i_1_n_0 ;
  wire \RX_data_a[14]_i_1_n_0 ;
  wire \RX_data_a[15]_i_1_n_0 ;
  wire \RX_data_a[16]_i_1_n_0 ;
  wire \RX_data_a[17]_i_1_n_0 ;
  wire \RX_data_a[18]_i_1_n_0 ;
  wire \RX_data_a[19]_i_1_n_0 ;
  wire \RX_data_a[1]_i_1_n_0 ;
  wire \RX_data_a[20]_i_1_n_0 ;
  wire \RX_data_a[21]_i_1_n_0 ;
  wire \RX_data_a[22]_i_1_n_0 ;
  wire \RX_data_a[23]_i_1_n_0 ;
  wire \RX_data_a[24]_i_1_n_0 ;
  wire \RX_data_a[25]_i_1_n_0 ;
  wire \RX_data_a[26]_i_1_n_0 ;
  wire \RX_data_a[27]_i_1_n_0 ;
  wire \RX_data_a[28]_i_1_n_0 ;
  wire \RX_data_a[29]_i_1_n_0 ;
  wire \RX_data_a[2]_i_1_n_0 ;
  wire \RX_data_a[30]_i_1_n_0 ;
  wire \RX_data_a[31]_i_1_n_0 ;
  wire \RX_data_a[3]_i_1_n_0 ;
  wire \RX_data_a[4]_i_1_n_0 ;
  wire \RX_data_a[5]_i_1_n_0 ;
  wire \RX_data_a[6]_i_1_n_0 ;
  wire \RX_data_a[7]_i_1_n_0 ;
  wire \RX_data_a[8]_i_1_n_0 ;
  wire \RX_data_a[9]_i_1_n_0 ;
  wire [31:4]RX_data_a_from_lms;
  wire [31:0]RX_data_b;
  wire [0:0]RX_data_b0;
  wire \RX_data_b[0]_i_1_n_0 ;
  wire \RX_data_b[10]_i_1_n_0 ;
  wire \RX_data_b[11]_i_1_n_0 ;
  wire \RX_data_b[12]_i_1_n_0 ;
  wire \RX_data_b[13]_i_1_n_0 ;
  wire \RX_data_b[14]_i_1_n_0 ;
  wire \RX_data_b[15]_i_1_n_0 ;
  wire \RX_data_b[16]_i_1_n_0 ;
  wire \RX_data_b[17]_i_1_n_0 ;
  wire \RX_data_b[18]_i_1_n_0 ;
  wire \RX_data_b[19]_i_1_n_0 ;
  wire \RX_data_b[1]_i_1_n_0 ;
  wire \RX_data_b[20]_i_1_n_0 ;
  wire \RX_data_b[21]_i_1_n_0 ;
  wire \RX_data_b[22]_i_1_n_0 ;
  wire \RX_data_b[23]_i_1_n_0 ;
  wire \RX_data_b[24]_i_1_n_0 ;
  wire \RX_data_b[25]_i_1_n_0 ;
  wire \RX_data_b[26]_i_1_n_0 ;
  wire \RX_data_b[27]_i_1_n_0 ;
  wire \RX_data_b[28]_i_1_n_0 ;
  wire \RX_data_b[29]_i_1_n_0 ;
  wire \RX_data_b[2]_i_1_n_0 ;
  wire \RX_data_b[30]_i_1_n_0 ;
  wire \RX_data_b[31]_i_1_n_0 ;
  wire \RX_data_b[3]_i_1_n_0 ;
  wire \RX_data_b[4]_i_1_n_0 ;
  wire \RX_data_b[5]_i_1_n_0 ;
  wire \RX_data_b[6]_i_1_n_0 ;
  wire \RX_data_b[7]_i_1_n_0 ;
  wire \RX_data_b[8]_i_1_n_0 ;
  wire \RX_data_b[9]_i_1_n_0 ;
  wire [31:4]RX_data_b_from_lms;
  wire [31:0]RX_pattern_reg;
  wire \RX_pattern_reg[0]_i_1_n_0 ;
  wire \RX_pattern_reg[0]_i_1_n_1 ;
  wire \RX_pattern_reg[0]_i_1_n_2 ;
  wire \RX_pattern_reg[0]_i_1_n_3 ;
  wire \RX_pattern_reg[0]_i_1_n_4 ;
  wire \RX_pattern_reg[0]_i_1_n_5 ;
  wire \RX_pattern_reg[0]_i_1_n_6 ;
  wire \RX_pattern_reg[0]_i_1_n_7 ;
  wire \RX_pattern_reg[12]_i_1_n_0 ;
  wire \RX_pattern_reg[12]_i_1_n_1 ;
  wire \RX_pattern_reg[12]_i_1_n_2 ;
  wire \RX_pattern_reg[12]_i_1_n_3 ;
  wire \RX_pattern_reg[12]_i_1_n_4 ;
  wire \RX_pattern_reg[12]_i_1_n_5 ;
  wire \RX_pattern_reg[12]_i_1_n_6 ;
  wire \RX_pattern_reg[12]_i_1_n_7 ;
  wire \RX_pattern_reg[16]_i_1_n_0 ;
  wire \RX_pattern_reg[16]_i_1_n_1 ;
  wire \RX_pattern_reg[16]_i_1_n_2 ;
  wire \RX_pattern_reg[16]_i_1_n_3 ;
  wire \RX_pattern_reg[16]_i_1_n_4 ;
  wire \RX_pattern_reg[16]_i_1_n_5 ;
  wire \RX_pattern_reg[16]_i_1_n_6 ;
  wire \RX_pattern_reg[16]_i_1_n_7 ;
  wire \RX_pattern_reg[20]_i_1_n_0 ;
  wire \RX_pattern_reg[20]_i_1_n_1 ;
  wire \RX_pattern_reg[20]_i_1_n_2 ;
  wire \RX_pattern_reg[20]_i_1_n_3 ;
  wire \RX_pattern_reg[20]_i_1_n_4 ;
  wire \RX_pattern_reg[20]_i_1_n_5 ;
  wire \RX_pattern_reg[20]_i_1_n_6 ;
  wire \RX_pattern_reg[20]_i_1_n_7 ;
  wire \RX_pattern_reg[24]_i_1_n_0 ;
  wire \RX_pattern_reg[24]_i_1_n_1 ;
  wire \RX_pattern_reg[24]_i_1_n_2 ;
  wire \RX_pattern_reg[24]_i_1_n_3 ;
  wire \RX_pattern_reg[24]_i_1_n_4 ;
  wire \RX_pattern_reg[24]_i_1_n_5 ;
  wire \RX_pattern_reg[24]_i_1_n_6 ;
  wire \RX_pattern_reg[24]_i_1_n_7 ;
  wire \RX_pattern_reg[28]_i_1_n_1 ;
  wire \RX_pattern_reg[28]_i_1_n_2 ;
  wire \RX_pattern_reg[28]_i_1_n_3 ;
  wire \RX_pattern_reg[28]_i_1_n_4 ;
  wire \RX_pattern_reg[28]_i_1_n_5 ;
  wire \RX_pattern_reg[28]_i_1_n_6 ;
  wire \RX_pattern_reg[28]_i_1_n_7 ;
  wire \RX_pattern_reg[4]_i_1_n_0 ;
  wire \RX_pattern_reg[4]_i_1_n_1 ;
  wire \RX_pattern_reg[4]_i_1_n_2 ;
  wire \RX_pattern_reg[4]_i_1_n_3 ;
  wire \RX_pattern_reg[4]_i_1_n_4 ;
  wire \RX_pattern_reg[4]_i_1_n_5 ;
  wire \RX_pattern_reg[4]_i_1_n_6 ;
  wire \RX_pattern_reg[4]_i_1_n_7 ;
  wire \RX_pattern_reg[8]_i_1_n_0 ;
  wire \RX_pattern_reg[8]_i_1_n_1 ;
  wire \RX_pattern_reg[8]_i_1_n_2 ;
  wire \RX_pattern_reg[8]_i_1_n_3 ;
  wire \RX_pattern_reg[8]_i_1_n_4 ;
  wire \RX_pattern_reg[8]_i_1_n_5 ;
  wire \RX_pattern_reg[8]_i_1_n_6 ;
  wire \RX_pattern_reg[8]_i_1_n_7 ;
  wire SYS_clk;
  wire [31:0]SYS_config_data;
  wire SYS_config_write;
  wire [31:0]SYS_data_clk_counter;
  wire [15:0]SYS_prbs_ctrl;
  wire [63:0]SYS_prbs_e;
  wire [31:0]SYS_prbs_stat;
  wire SYS_rst;
  wire [31:0]SYS_test_data_a_rx;
  wire [31:0]SYS_test_data_a_tx;
  wire [31:0]SYS_test_data_b_rx;
  wire [31:0]SYS_test_data_b_tx;
  wire [4:0]SYS_test_mode;
  wire SYS_test_mode0;
  wire [31:0]TX_data_a;
  wire [31:0]TX_data_a_to_lms;
  wire \TX_data_a_to_lms[0]_i_1_n_0 ;
  wire \TX_data_a_to_lms[0]_i_2_n_0 ;
  wire \TX_data_a_to_lms[10]_i_1_n_0 ;
  wire \TX_data_a_to_lms[10]_i_2_n_0 ;
  wire \TX_data_a_to_lms[11]_i_1_n_0 ;
  wire \TX_data_a_to_lms[11]_i_2_n_0 ;
  wire \TX_data_a_to_lms[12]_i_1_n_0 ;
  wire \TX_data_a_to_lms[12]_i_2_n_0 ;
  wire \TX_data_a_to_lms[13]_i_1_n_0 ;
  wire \TX_data_a_to_lms[13]_i_2_n_0 ;
  wire \TX_data_a_to_lms[14]_i_1_n_0 ;
  wire \TX_data_a_to_lms[14]_i_2_n_0 ;
  wire \TX_data_a_to_lms[15]_i_1_n_0 ;
  wire \TX_data_a_to_lms[15]_i_2_n_0 ;
  wire \TX_data_a_to_lms[16]_i_1_n_0 ;
  wire \TX_data_a_to_lms[16]_i_2_n_0 ;
  wire \TX_data_a_to_lms[17]_i_1_n_0 ;
  wire \TX_data_a_to_lms[17]_i_2_n_0 ;
  wire \TX_data_a_to_lms[18]_i_1_n_0 ;
  wire \TX_data_a_to_lms[18]_i_2_n_0 ;
  wire \TX_data_a_to_lms[19]_i_1_n_0 ;
  wire \TX_data_a_to_lms[19]_i_2_n_0 ;
  wire \TX_data_a_to_lms[1]_i_1_n_0 ;
  wire \TX_data_a_to_lms[1]_i_2_n_0 ;
  wire \TX_data_a_to_lms[20]_i_1_n_0 ;
  wire \TX_data_a_to_lms[20]_i_2_n_0 ;
  wire \TX_data_a_to_lms[21]_i_1_n_0 ;
  wire \TX_data_a_to_lms[21]_i_2_n_0 ;
  wire \TX_data_a_to_lms[22]_i_1_n_0 ;
  wire \TX_data_a_to_lms[22]_i_2_n_0 ;
  wire \TX_data_a_to_lms[23]_i_1_n_0 ;
  wire \TX_data_a_to_lms[23]_i_2_n_0 ;
  wire \TX_data_a_to_lms[24]_i_1_n_0 ;
  wire \TX_data_a_to_lms[24]_i_2_n_0 ;
  wire \TX_data_a_to_lms[25]_i_1_n_0 ;
  wire \TX_data_a_to_lms[25]_i_2_n_0 ;
  wire \TX_data_a_to_lms[26]_i_1_n_0 ;
  wire \TX_data_a_to_lms[26]_i_2_n_0 ;
  wire \TX_data_a_to_lms[27]_i_1_n_0 ;
  wire \TX_data_a_to_lms[27]_i_2_n_0 ;
  wire \TX_data_a_to_lms[28]_i_1_n_0 ;
  wire \TX_data_a_to_lms[28]_i_2_n_0 ;
  wire \TX_data_a_to_lms[29]_i_1_n_0 ;
  wire \TX_data_a_to_lms[29]_i_2_n_0 ;
  wire \TX_data_a_to_lms[2]_i_1_n_0 ;
  wire \TX_data_a_to_lms[2]_i_2_n_0 ;
  wire \TX_data_a_to_lms[30]_i_1_n_0 ;
  wire \TX_data_a_to_lms[30]_i_2_n_0 ;
  wire \TX_data_a_to_lms[31]_i_1_n_0 ;
  wire \TX_data_a_to_lms[31]_i_2_n_0 ;
  wire \TX_data_a_to_lms[3]_i_1_n_0 ;
  wire \TX_data_a_to_lms[3]_i_2_n_0 ;
  wire \TX_data_a_to_lms[4]_i_1_n_0 ;
  wire \TX_data_a_to_lms[4]_i_2_n_0 ;
  wire \TX_data_a_to_lms[5]_i_1_n_0 ;
  wire \TX_data_a_to_lms[5]_i_2_n_0 ;
  wire \TX_data_a_to_lms[6]_i_1_n_0 ;
  wire \TX_data_a_to_lms[6]_i_2_n_0 ;
  wire \TX_data_a_to_lms[7]_i_1_n_0 ;
  wire \TX_data_a_to_lms[7]_i_2_n_0 ;
  wire \TX_data_a_to_lms[8]_i_1_n_0 ;
  wire \TX_data_a_to_lms[8]_i_2_n_0 ;
  wire \TX_data_a_to_lms[9]_i_1_n_0 ;
  wire \TX_data_a_to_lms[9]_i_2_n_0 ;
  wire [31:0]TX_data_b;
  wire [31:0]TX_data_b_to_lms;
  wire \TX_data_b_to_lms[0]_i_1_n_0 ;
  wire \TX_data_b_to_lms[0]_i_2_n_0 ;
  wire \TX_data_b_to_lms[10]_i_1_n_0 ;
  wire \TX_data_b_to_lms[10]_i_2_n_0 ;
  wire \TX_data_b_to_lms[11]_i_1_n_0 ;
  wire \TX_data_b_to_lms[11]_i_2_n_0 ;
  wire \TX_data_b_to_lms[12]_i_1_n_0 ;
  wire \TX_data_b_to_lms[12]_i_2_n_0 ;
  wire \TX_data_b_to_lms[13]_i_1_n_0 ;
  wire \TX_data_b_to_lms[13]_i_2_n_0 ;
  wire \TX_data_b_to_lms[14]_i_1_n_0 ;
  wire \TX_data_b_to_lms[14]_i_2_n_0 ;
  wire \TX_data_b_to_lms[15]_i_1_n_0 ;
  wire \TX_data_b_to_lms[15]_i_2_n_0 ;
  wire \TX_data_b_to_lms[16]_i_1_n_0 ;
  wire \TX_data_b_to_lms[16]_i_2_n_0 ;
  wire \TX_data_b_to_lms[17]_i_1_n_0 ;
  wire \TX_data_b_to_lms[17]_i_2_n_0 ;
  wire \TX_data_b_to_lms[18]_i_1_n_0 ;
  wire \TX_data_b_to_lms[18]_i_2_n_0 ;
  wire \TX_data_b_to_lms[19]_i_1_n_0 ;
  wire \TX_data_b_to_lms[19]_i_2_n_0 ;
  wire \TX_data_b_to_lms[1]_i_1_n_0 ;
  wire \TX_data_b_to_lms[1]_i_2_n_0 ;
  wire \TX_data_b_to_lms[20]_i_1_n_0 ;
  wire \TX_data_b_to_lms[20]_i_2_n_0 ;
  wire \TX_data_b_to_lms[21]_i_1_n_0 ;
  wire \TX_data_b_to_lms[21]_i_2_n_0 ;
  wire \TX_data_b_to_lms[22]_i_1_n_0 ;
  wire \TX_data_b_to_lms[22]_i_2_n_0 ;
  wire \TX_data_b_to_lms[23]_i_1_n_0 ;
  wire \TX_data_b_to_lms[23]_i_2_n_0 ;
  wire \TX_data_b_to_lms[24]_i_1_n_0 ;
  wire \TX_data_b_to_lms[24]_i_2_n_0 ;
  wire \TX_data_b_to_lms[25]_i_1_n_0 ;
  wire \TX_data_b_to_lms[25]_i_2_n_0 ;
  wire \TX_data_b_to_lms[26]_i_1_n_0 ;
  wire \TX_data_b_to_lms[26]_i_2_n_0 ;
  wire \TX_data_b_to_lms[27]_i_1_n_0 ;
  wire \TX_data_b_to_lms[27]_i_2_n_0 ;
  wire \TX_data_b_to_lms[28]_i_1_n_0 ;
  wire \TX_data_b_to_lms[28]_i_2_n_0 ;
  wire \TX_data_b_to_lms[29]_i_1_n_0 ;
  wire \TX_data_b_to_lms[29]_i_2_n_0 ;
  wire \TX_data_b_to_lms[2]_i_1_n_0 ;
  wire \TX_data_b_to_lms[2]_i_2_n_0 ;
  wire \TX_data_b_to_lms[30]_i_1_n_0 ;
  wire \TX_data_b_to_lms[30]_i_2_n_0 ;
  wire \TX_data_b_to_lms[31]_i_1_n_0 ;
  wire \TX_data_b_to_lms[31]_i_2_n_0 ;
  wire \TX_data_b_to_lms[3]_i_1_n_0 ;
  wire \TX_data_b_to_lms[3]_i_2_n_0 ;
  wire \TX_data_b_to_lms[4]_i_1_n_0 ;
  wire \TX_data_b_to_lms[4]_i_2_n_0 ;
  wire \TX_data_b_to_lms[5]_i_1_n_0 ;
  wire \TX_data_b_to_lms[5]_i_2_n_0 ;
  wire \TX_data_b_to_lms[6]_i_1_n_0 ;
  wire \TX_data_b_to_lms[6]_i_2_n_0 ;
  wire \TX_data_b_to_lms[7]_i_1_n_0 ;
  wire \TX_data_b_to_lms[7]_i_2_n_0 ;
  wire \TX_data_b_to_lms[8]_i_1_n_0 ;
  wire \TX_data_b_to_lms[8]_i_2_n_0 ;
  wire \TX_data_b_to_lms[9]_i_1_n_0 ;
  wire \TX_data_b_to_lms[9]_i_2_n_0 ;
  wire [31:0]TX_test_data_b;
  wire \count[6]_i_2_n_0 ;
  wire [7:0]counterF;
  wire [7:2]counterFMax;
  wire \counterFMax[7]_i_1_n_0 ;
  wire \counterF[0]_i_1_n_0 ;
  wire \counterF[1]_i_1_n_0 ;
  wire \counterF[2]_i_1_n_0 ;
  wire \counterF[3]_i_1_n_0 ;
  wire \counterF[4]_i_1_n_0 ;
  wire \counterF[5]_i_1_n_0 ;
  wire \counterF[6]_i_1_n_0 ;
  wire \counterF[7]_i_1_n_0 ;
  wire \counterF[7]_i_2_n_0 ;
  wire [7:0]counterR;
  wire [7:2]counterRMax;
  wire \counterRMax[7]_i_1_n_0 ;
  wire \counterR[7]_i_2_n_0 ;
  wire \counter_src[0]_i_2_n_0 ;
  wire \counter_src_reg[0]_i_1_n_0 ;
  wire \counter_src_reg[0]_i_1_n_1 ;
  wire \counter_src_reg[0]_i_1_n_2 ;
  wire \counter_src_reg[0]_i_1_n_3 ;
  wire \counter_src_reg[0]_i_1_n_4 ;
  wire \counter_src_reg[0]_i_1_n_5 ;
  wire \counter_src_reg[0]_i_1_n_6 ;
  wire \counter_src_reg[0]_i_1_n_7 ;
  wire \counter_src_reg[12]_i_1_n_0 ;
  wire \counter_src_reg[12]_i_1_n_1 ;
  wire \counter_src_reg[12]_i_1_n_2 ;
  wire \counter_src_reg[12]_i_1_n_3 ;
  wire \counter_src_reg[12]_i_1_n_4 ;
  wire \counter_src_reg[12]_i_1_n_5 ;
  wire \counter_src_reg[12]_i_1_n_6 ;
  wire \counter_src_reg[12]_i_1_n_7 ;
  wire \counter_src_reg[16]_i_1_n_0 ;
  wire \counter_src_reg[16]_i_1_n_1 ;
  wire \counter_src_reg[16]_i_1_n_2 ;
  wire \counter_src_reg[16]_i_1_n_3 ;
  wire \counter_src_reg[16]_i_1_n_4 ;
  wire \counter_src_reg[16]_i_1_n_5 ;
  wire \counter_src_reg[16]_i_1_n_6 ;
  wire \counter_src_reg[16]_i_1_n_7 ;
  wire \counter_src_reg[20]_i_1_n_0 ;
  wire \counter_src_reg[20]_i_1_n_1 ;
  wire \counter_src_reg[20]_i_1_n_2 ;
  wire \counter_src_reg[20]_i_1_n_3 ;
  wire \counter_src_reg[20]_i_1_n_4 ;
  wire \counter_src_reg[20]_i_1_n_5 ;
  wire \counter_src_reg[20]_i_1_n_6 ;
  wire \counter_src_reg[20]_i_1_n_7 ;
  wire \counter_src_reg[24]_i_1_n_0 ;
  wire \counter_src_reg[24]_i_1_n_1 ;
  wire \counter_src_reg[24]_i_1_n_2 ;
  wire \counter_src_reg[24]_i_1_n_3 ;
  wire \counter_src_reg[24]_i_1_n_4 ;
  wire \counter_src_reg[24]_i_1_n_5 ;
  wire \counter_src_reg[24]_i_1_n_6 ;
  wire \counter_src_reg[24]_i_1_n_7 ;
  wire \counter_src_reg[28]_i_1_n_1 ;
  wire \counter_src_reg[28]_i_1_n_2 ;
  wire \counter_src_reg[28]_i_1_n_3 ;
  wire \counter_src_reg[28]_i_1_n_4 ;
  wire \counter_src_reg[28]_i_1_n_5 ;
  wire \counter_src_reg[28]_i_1_n_6 ;
  wire \counter_src_reg[28]_i_1_n_7 ;
  wire \counter_src_reg[4]_i_1_n_0 ;
  wire \counter_src_reg[4]_i_1_n_1 ;
  wire \counter_src_reg[4]_i_1_n_2 ;
  wire \counter_src_reg[4]_i_1_n_3 ;
  wire \counter_src_reg[4]_i_1_n_4 ;
  wire \counter_src_reg[4]_i_1_n_5 ;
  wire \counter_src_reg[4]_i_1_n_6 ;
  wire \counter_src_reg[4]_i_1_n_7 ;
  wire \counter_src_reg[8]_i_1_n_0 ;
  wire \counter_src_reg[8]_i_1_n_1 ;
  wire \counter_src_reg[8]_i_1_n_2 ;
  wire \counter_src_reg[8]_i_1_n_3 ;
  wire \counter_src_reg[8]_i_1_n_4 ;
  wire \counter_src_reg[8]_i_1_n_5 ;
  wire \counter_src_reg[8]_i_1_n_6 ;
  wire \counter_src_reg[8]_i_1_n_7 ;
  wire data_clk_2x;
  wire data_stb_i_i_1_n_0;
  wire \err[10]_i_1_n_0 ;
  wire \err[11]_i_1_n_0 ;
  wire \err[12]_i_1_n_0 ;
  wire \err[13]_i_1_n_0 ;
  wire \err[14]_i_1_n_0 ;
  wire \err[15]_i_1_n_0 ;
  wire \err[20]_i_1_n_0 ;
  wire \err[21]_i_1_n_0 ;
  wire \err[22]_i_1_n_0 ;
  wire \err[23]_i_1_n_0 ;
  wire \err[24]_i_1_n_0 ;
  wire \err[25]_i_1_n_0 ;
  wire \err[26]_i_1_n_0 ;
  wire \err[27]_i_1_n_0 ;
  wire \err[28]_i_1_n_0 ;
  wire \err[29]_i_1_n_0 ;
  wire \err[30]_i_1_n_0 ;
  wire \err[31]_i_1_n_0 ;
  wire \err[36]_i_1_n_0 ;
  wire \err[37]_i_1_n_0 ;
  wire \err[38]_i_1_n_0 ;
  wire \err[39]_i_1_n_0 ;
  wire \err[40]_i_1_n_0 ;
  wire \err[41]_i_1_n_0 ;
  wire \err[42]_i_1_n_0 ;
  wire \err[43]_i_1_n_0 ;
  wire \err[44]_i_1_n_0 ;
  wire \err[45]_i_1_n_0 ;
  wire \err[46]_i_1_n_0 ;
  wire \err[47]_i_1_n_0 ;
  wire \err[4]_i_1_n_0 ;
  wire \err[52]_i_1_n_0 ;
  wire \err[53]_i_1_n_0 ;
  wire \err[54]_i_1_n_0 ;
  wire \err[55]_i_1_n_0 ;
  wire \err[56]_i_1_n_0 ;
  wire \err[57]_i_1_n_0 ;
  wire \err[58]_i_1_n_0 ;
  wire \err[59]_i_1_n_0 ;
  wire \err[5]_i_1_n_0 ;
  wire \err[60]_i_1_n_0 ;
  wire \err[61]_i_1_n_0 ;
  wire \err[62]_i_1_n_0 ;
  wire \err[63]_i_1_n_0 ;
  wire \err[63]_i_2_n_0 ;
  wire \err[6]_i_1_n_0 ;
  wire \err[7]_i_1_n_0 ;
  wire \err[8]_i_1_n_0 ;
  wire \err[9]_i_1_n_0 ;
  wire \out[63]_i_1_n_0 ;
  wire \out_reg[2]_srl3___u_tester_u_chk_out_reg_r_1_i_1_n_0 ;
  wire [5:5]p_0_in;
  wire [5:5]p_0_in0_in;
  wire [6:0]p_0_in__0;
  wire [7:0]p_0_in__1;
  wire [3:0]p_0_in__2;
  wire [0:0]p_0_out;
  wire [7:0]plusOp;
  wire [15:0]prbs_ctrl;
  wire [63:4]prbs_e;
  wire [27:0]prbs_stat;
  wire rx_err0_carry__0_i_1_n_0;
  wire rx_err0_carry__0_i_2_n_0;
  wire rx_err0_carry__0_i_3_n_0;
  wire rx_err0_carry__0_i_4_n_0;
  wire rx_err0_carry__1_i_1_n_0;
  wire rx_err0_carry__1_i_2_n_0;
  wire rx_err0_carry__1_i_3_n_0;
  wire rx_err0_carry__1_i_4_n_0;
  wire rx_err0_carry__2_i_1_n_0;
  wire rx_err0_carry__2_i_2_n_0;
  wire rx_err0_carry__2_i_3_n_0;
  wire rx_err0_carry__2_i_4_n_0;
  wire rx_err0_carry__3_i_1_n_0;
  wire rx_err0_carry__3_i_2_n_0;
  wire rx_err0_carry__3_i_3_n_0;
  wire rx_err0_carry_i_1_n_0;
  wire rx_err0_carry_i_2_n_0;
  wire rx_err0_carry_i_3_n_0;
  wire rx_err0_carry_i_4_n_0;
  wire \rx_err_cnt[7]_i_3_n_0 ;
  wire rx_err_s_i_1_n_0;
  wire rx_test_mode_xfer_n_0;
  wire rx_test_mode_xfer_n_1;
  wire samp_clk_1x_i_1_n_0;
  wire samp_clk_2x_i_10_n_0;
  wire samp_clk_2x_i_13_n_0;
  wire samp_clk_2x_i_14_n_0;
  wire samp_clk_2x_i_1_n_0;
  wire samp_clk_2x_i_2_n_0;
  wire samp_clk_2x_i_3_n_0;
  wire samp_clk_2x_i_4_n_0;
  wire samp_clk_2x_i_5_n_0;
  wire samp_clk_2x_i_6_n_0;
  wire samp_clk_2x_i_7_n_0;
  wire samp_clk_2x_i_8_n_0;
  wire samp_clk_2x_i_9_n_0;
  wire samp_clk_fb_i_1_n_0;
  wire sel_mmcm;
  wire sel_mmcm_i_1_n_0;
  wire \slip_cnt[3]_i_1_n_0 ;
  wire \st_cur[0]_i_1_n_0 ;
  wire \st_cur[0]_i_2_n_0 ;
  wire \st_cur[1]_i_10_n_0 ;
  wire \st_cur[1]_i_11_n_0 ;
  wire \st_cur[1]_i_12_n_0 ;
  wire \st_cur[1]_i_13_n_0 ;
  wire \st_cur[1]_i_14_n_0 ;
  wire \st_cur[1]_i_15_n_0 ;
  wire \st_cur[1]_i_16_n_0 ;
  wire \st_cur[1]_i_1_n_0 ;
  wire \st_cur[1]_i_2_n_0 ;
  wire \st_cur[1]_i_4_n_0 ;
  wire \st_cur[1]_i_5_n_0 ;
  wire \st_cur[1]_i_6_n_0 ;
  wire \st_cur[1]_i_7_n_0 ;
  wire \st_cur[1]_i_8_n_0 ;
  wire \st_cur[1]_i_9_n_0 ;
  wire [11:0]\trxiq_rx/data1 ;
  wire [11:0]\trxiq_rx/data1_reg ;
  wire [11:0]\trxiq_rx/data2 ;
  wire [11:0]\trxiq_rx/data2_reg ;
  wire \trxiq_rx/iddr_sel_n_0 ;
  wire \trxiq_tx/D1 ;
  wire \trxiq_tx/D2 ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[10] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[11] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[12] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[13] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[14] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[15] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[20] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[21] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[22] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[23] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[24] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[25] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[26] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[27] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[28] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[29] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[30] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[31] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[4] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[5] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[6] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[7] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[8] ;
  wire \trxiq_tx/data_a_reg_reg_n_0_[9] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[10] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[11] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[12] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[13] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[14] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[15] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[20] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[21] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[22] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[23] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[24] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[25] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[26] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[27] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[28] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[29] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[30] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[31] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[4] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[5] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[6] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[7] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[8] ;
  wire \trxiq_tx/data_b_reg_reg_n_0_[9] ;
  wire \trxiq_tx/data_stb_i_reg_n_0 ;
  wire \trxiq_tx/gen_diq_bits[0].oddr_data_i_1_n_0 ;
  wire \trxiq_tx/gen_diq_bits[0].oddr_data_i_2_n_0 ;
  wire \trxiq_tx/gen_diq_bits[10].oddr_data_i_1_n_0 ;
  wire \trxiq_tx/gen_diq_bits[10].oddr_data_i_2_n_0 ;
  wire \trxiq_tx/gen_diq_bits[1].oddr_data_i_1_n_0 ;
  wire \trxiq_tx/gen_diq_bits[1].oddr_data_i_2_n_0 ;
  wire \trxiq_tx/gen_diq_bits[2].oddr_data_i_1_n_0 ;
  wire \trxiq_tx/gen_diq_bits[2].oddr_data_i_2_n_0 ;
  wire \trxiq_tx/gen_diq_bits[3].oddr_data_i_1_n_0 ;
  wire \trxiq_tx/gen_diq_bits[3].oddr_data_i_2_n_0 ;
  wire \trxiq_tx/gen_diq_bits[4].oddr_data_i_1_n_0 ;
  wire \trxiq_tx/gen_diq_bits[4].oddr_data_i_2_n_0 ;
  wire \trxiq_tx/gen_diq_bits[5].oddr_data_i_1_n_0 ;
  wire \trxiq_tx/gen_diq_bits[5].oddr_data_i_2_n_0 ;
  wire \trxiq_tx/gen_diq_bits[6].oddr_data_i_1_n_0 ;
  wire \trxiq_tx/gen_diq_bits[6].oddr_data_i_2_n_0 ;
  wire \trxiq_tx/gen_diq_bits[7].oddr_data_i_1_n_0 ;
  wire \trxiq_tx/gen_diq_bits[7].oddr_data_i_2_n_0 ;
  wire \trxiq_tx/gen_diq_bits[8].oddr_data_i_1_n_0 ;
  wire \trxiq_tx/gen_diq_bits[8].oddr_data_i_2_n_0 ;
  wire \trxiq_tx/gen_diq_bits[9].oddr_data_i_1_n_0 ;
  wire \trxiq_tx/gen_diq_bits[9].oddr_data_i_2_n_0 ;
  wire \tx_cnt[0]_i_3_n_0 ;
  wire \tx_cnt_reg[0]_i_2_n_0 ;
  wire \tx_cnt_reg[0]_i_2_n_1 ;
  wire \tx_cnt_reg[0]_i_2_n_2 ;
  wire \tx_cnt_reg[0]_i_2_n_3 ;
  wire \tx_cnt_reg[0]_i_2_n_4 ;
  wire \tx_cnt_reg[0]_i_2_n_5 ;
  wire \tx_cnt_reg[0]_i_2_n_6 ;
  wire \tx_cnt_reg[0]_i_2_n_7 ;
  wire \tx_cnt_reg[12]_i_1_n_0 ;
  wire \tx_cnt_reg[12]_i_1_n_1 ;
  wire \tx_cnt_reg[12]_i_1_n_2 ;
  wire \tx_cnt_reg[12]_i_1_n_3 ;
  wire \tx_cnt_reg[12]_i_1_n_4 ;
  wire \tx_cnt_reg[12]_i_1_n_5 ;
  wire \tx_cnt_reg[12]_i_1_n_6 ;
  wire \tx_cnt_reg[12]_i_1_n_7 ;
  wire \tx_cnt_reg[16]_i_1_n_0 ;
  wire \tx_cnt_reg[16]_i_1_n_1 ;
  wire \tx_cnt_reg[16]_i_1_n_2 ;
  wire \tx_cnt_reg[16]_i_1_n_3 ;
  wire \tx_cnt_reg[16]_i_1_n_4 ;
  wire \tx_cnt_reg[16]_i_1_n_5 ;
  wire \tx_cnt_reg[16]_i_1_n_6 ;
  wire \tx_cnt_reg[16]_i_1_n_7 ;
  wire \tx_cnt_reg[20]_i_1_n_0 ;
  wire \tx_cnt_reg[20]_i_1_n_1 ;
  wire \tx_cnt_reg[20]_i_1_n_2 ;
  wire \tx_cnt_reg[20]_i_1_n_3 ;
  wire \tx_cnt_reg[20]_i_1_n_4 ;
  wire \tx_cnt_reg[20]_i_1_n_5 ;
  wire \tx_cnt_reg[20]_i_1_n_6 ;
  wire \tx_cnt_reg[20]_i_1_n_7 ;
  wire \tx_cnt_reg[24]_i_1_n_0 ;
  wire \tx_cnt_reg[24]_i_1_n_1 ;
  wire \tx_cnt_reg[24]_i_1_n_2 ;
  wire \tx_cnt_reg[24]_i_1_n_3 ;
  wire \tx_cnt_reg[24]_i_1_n_4 ;
  wire \tx_cnt_reg[24]_i_1_n_5 ;
  wire \tx_cnt_reg[24]_i_1_n_6 ;
  wire \tx_cnt_reg[24]_i_1_n_7 ;
  wire \tx_cnt_reg[28]_i_1_n_2 ;
  wire \tx_cnt_reg[28]_i_1_n_3 ;
  wire \tx_cnt_reg[28]_i_1_n_5 ;
  wire \tx_cnt_reg[28]_i_1_n_6 ;
  wire \tx_cnt_reg[28]_i_1_n_7 ;
  wire \tx_cnt_reg[4]_i_1_n_0 ;
  wire \tx_cnt_reg[4]_i_1_n_1 ;
  wire \tx_cnt_reg[4]_i_1_n_2 ;
  wire \tx_cnt_reg[4]_i_1_n_3 ;
  wire \tx_cnt_reg[4]_i_1_n_4 ;
  wire \tx_cnt_reg[4]_i_1_n_5 ;
  wire \tx_cnt_reg[4]_i_1_n_6 ;
  wire \tx_cnt_reg[4]_i_1_n_7 ;
  wire \tx_cnt_reg[8]_i_1_n_0 ;
  wire \tx_cnt_reg[8]_i_1_n_1 ;
  wire \tx_cnt_reg[8]_i_1_n_2 ;
  wire \tx_cnt_reg[8]_i_1_n_3 ;
  wire \tx_cnt_reg[8]_i_1_n_4 ;
  wire \tx_cnt_reg[8]_i_1_n_5 ;
  wire \tx_cnt_reg[8]_i_1_n_6 ;
  wire \tx_cnt_reg[8]_i_1_n_7 ;
  wire [31:0]\tx_cycle_counter/counter_src_reg ;
  wire tx_test_data_xfer_n_32;
  wire tx_test_data_xfer_n_33;
  wire tx_test_data_xfer_n_34;
  wire tx_test_data_xfer_n_35;
  wire tx_test_data_xfer_n_36;
  wire tx_test_data_xfer_n_37;
  wire tx_test_data_xfer_n_38;
  wire tx_test_data_xfer_n_39;
  wire tx_test_data_xfer_n_40;
  wire tx_test_data_xfer_n_41;
  wire tx_test_data_xfer_n_42;
  wire tx_test_data_xfer_n_43;
  wire tx_test_data_xfer_n_44;
  wire tx_test_data_xfer_n_45;
  wire tx_test_data_xfer_n_46;
  wire tx_test_data_xfer_n_47;
  wire tx_test_data_xfer_n_48;
  wire tx_test_data_xfer_n_49;
  wire tx_test_data_xfer_n_50;
  wire tx_test_data_xfer_n_51;
  wire tx_test_data_xfer_n_52;
  wire tx_test_data_xfer_n_53;
  wire tx_test_data_xfer_n_54;
  wire tx_test_data_xfer_n_55;
  wire tx_test_data_xfer_n_56;
  wire tx_test_data_xfer_n_57;
  wire tx_test_data_xfer_n_58;
  wire tx_test_data_xfer_n_59;
  wire tx_test_data_xfer_n_60;
  wire tx_test_data_xfer_n_61;
  wire tx_test_data_xfer_n_62;
  wire tx_test_data_xfer_n_63;
  wire tx_test_mode_xfer_n_1;
  wire tx_test_mode_xfer_n_2;
  wire \u_clocks/CLKFBIN ;
  wire [1:0]\u_clocks/clk_hist ;
  wire \u_clocks/data_clk_fb ;
  wire \u_clocks/mmcm_clk_1x ;
  wire \u_clocks/mmcm_clk_2x ;
  wire \u_clocks/mmcm_clk_fb ;
  wire \u_clocks/mmcm_mclk_n_10 ;
  wire \u_clocks/mmcm_mclk_n_11 ;
  wire \u_clocks/mmcm_mclk_n_12 ;
  wire \u_clocks/mmcm_mclk_n_13 ;
  wire \u_clocks/mmcm_mclk_n_14 ;
  wire \u_clocks/mmcm_mclk_n_15 ;
  wire \u_clocks/mmcm_mclk_n_16 ;
  wire \u_clocks/mmcm_mclk_n_17 ;
  wire \u_clocks/mmcm_mclk_n_18 ;
  wire \u_clocks/mmcm_mclk_n_19 ;
  wire \u_clocks/mmcm_mclk_n_20 ;
  wire \u_clocks/mmcm_mclk_n_21 ;
  wire \u_clocks/mmcm_mclk_n_22 ;
  wire \u_clocks/mmcm_mclk_n_23 ;
  wire \u_clocks/mmcm_mclk_n_24 ;
  wire \u_clocks/mmcm_mclk_n_7 ;
  wire \u_clocks/mmcm_mclk_n_9 ;
  wire \u_clocks/samp_clk_1x ;
  wire \u_clocks/samp_clk_2x ;
  wire \u_clocks/samp_clk_fb_reg_n_0 ;
  wire \u_tester/chk_en ;
  wire [63:4]\u_tester/chk_out ;
  wire [6:0]\u_tester/count_reg__0 ;
  wire \u_tester/mask_count ;
  wire \u_tester/mask_count_reg[18]_srl3___u_tester_mask_count_reg_r_4_n_0 ;
  wire \u_tester/mask_count_reg[19]_u_tester_mask_count_reg_r_5_n_0 ;
  wire \u_tester/mask_count_reg[2]_srl3___u_tester_mask_count_reg_r_4_n_0 ;
  wire \u_tester/mask_count_reg[34]_srl3___u_tester_mask_count_reg_r_4_n_0 ;
  wire \u_tester/mask_count_reg[35]_u_tester_mask_count_reg_r_5_n_0 ;
  wire \u_tester/mask_count_reg[3]_u_tester_mask_count_reg_r_5_n_0 ;
  wire \u_tester/mask_count_reg[50]_srl3___u_tester_mask_count_reg_r_4_n_0 ;
  wire \u_tester/mask_count_reg[51]_u_tester_mask_count_reg_r_5_n_0 ;
  wire \u_tester/mask_count_reg_gate__0_n_0 ;
  wire \u_tester/mask_count_reg_gate__1_n_0 ;
  wire \u_tester/mask_count_reg_gate__2_n_0 ;
  wire \u_tester/mask_count_reg_gate_n_0 ;
  wire \u_tester/mask_count_reg_n_0_[10] ;
  wire \u_tester/mask_count_reg_n_0_[11] ;
  wire \u_tester/mask_count_reg_n_0_[12] ;
  wire \u_tester/mask_count_reg_n_0_[13] ;
  wire \u_tester/mask_count_reg_n_0_[14] ;
  wire \u_tester/mask_count_reg_n_0_[15] ;
  wire \u_tester/mask_count_reg_n_0_[20] ;
  wire \u_tester/mask_count_reg_n_0_[21] ;
  wire \u_tester/mask_count_reg_n_0_[22] ;
  wire \u_tester/mask_count_reg_n_0_[23] ;
  wire \u_tester/mask_count_reg_n_0_[24] ;
  wire \u_tester/mask_count_reg_n_0_[25] ;
  wire \u_tester/mask_count_reg_n_0_[26] ;
  wire \u_tester/mask_count_reg_n_0_[27] ;
  wire \u_tester/mask_count_reg_n_0_[28] ;
  wire \u_tester/mask_count_reg_n_0_[29] ;
  wire \u_tester/mask_count_reg_n_0_[30] ;
  wire \u_tester/mask_count_reg_n_0_[31] ;
  wire \u_tester/mask_count_reg_n_0_[36] ;
  wire \u_tester/mask_count_reg_n_0_[37] ;
  wire \u_tester/mask_count_reg_n_0_[38] ;
  wire \u_tester/mask_count_reg_n_0_[39] ;
  wire \u_tester/mask_count_reg_n_0_[40] ;
  wire \u_tester/mask_count_reg_n_0_[41] ;
  wire \u_tester/mask_count_reg_n_0_[42] ;
  wire \u_tester/mask_count_reg_n_0_[43] ;
  wire \u_tester/mask_count_reg_n_0_[44] ;
  wire \u_tester/mask_count_reg_n_0_[45] ;
  wire \u_tester/mask_count_reg_n_0_[46] ;
  wire \u_tester/mask_count_reg_n_0_[47] ;
  wire \u_tester/mask_count_reg_n_0_[4] ;
  wire \u_tester/mask_count_reg_n_0_[52] ;
  wire \u_tester/mask_count_reg_n_0_[53] ;
  wire \u_tester/mask_count_reg_n_0_[54] ;
  wire \u_tester/mask_count_reg_n_0_[55] ;
  wire \u_tester/mask_count_reg_n_0_[56] ;
  wire \u_tester/mask_count_reg_n_0_[57] ;
  wire \u_tester/mask_count_reg_n_0_[58] ;
  wire \u_tester/mask_count_reg_n_0_[59] ;
  wire \u_tester/mask_count_reg_n_0_[5] ;
  wire \u_tester/mask_count_reg_n_0_[60] ;
  wire \u_tester/mask_count_reg_n_0_[61] ;
  wire \u_tester/mask_count_reg_n_0_[62] ;
  wire \u_tester/mask_count_reg_n_0_[63] ;
  wire \u_tester/mask_count_reg_n_0_[6] ;
  wire \u_tester/mask_count_reg_n_0_[7] ;
  wire \u_tester/mask_count_reg_n_0_[8] ;
  wire \u_tester/mask_count_reg_n_0_[9] ;
  wire \u_tester/mask_count_reg_r_3_n_0 ;
  wire \u_tester/mask_count_reg_r_4_n_0 ;
  wire \u_tester/mask_count_reg_r_5_n_0 ;
  wire \u_tester/mask_count_reg_r_n_0 ;
  wire \u_tester/rx_err0 ;
  wire \u_tester/rx_err0_carry__0_n_0 ;
  wire \u_tester/rx_err0_carry__0_n_1 ;
  wire \u_tester/rx_err0_carry__0_n_2 ;
  wire \u_tester/rx_err0_carry__0_n_3 ;
  wire \u_tester/rx_err0_carry__1_n_0 ;
  wire \u_tester/rx_err0_carry__1_n_1 ;
  wire \u_tester/rx_err0_carry__1_n_2 ;
  wire \u_tester/rx_err0_carry__1_n_3 ;
  wire \u_tester/rx_err0_carry__2_n_0 ;
  wire \u_tester/rx_err0_carry__2_n_1 ;
  wire \u_tester/rx_err0_carry__2_n_2 ;
  wire \u_tester/rx_err0_carry__2_n_3 ;
  wire \u_tester/rx_err0_carry__3_n_2 ;
  wire \u_tester/rx_err0_carry__3_n_3 ;
  wire \u_tester/rx_err0_carry_n_0 ;
  wire \u_tester/rx_err0_carry_n_1 ;
  wire \u_tester/rx_err0_carry_n_2 ;
  wire \u_tester/rx_err0_carry_n_3 ;
  wire \u_tester/rx_err_cnt0 ;
  wire \u_tester/rx_slip ;
  wire [1:0]\u_tester/st_cur ;
  wire [1:1]\u_tester/st_nxt ;
  wire \u_tester/tx_cnt0 ;
  wire \u_tester/tx_cnt_reg_n_0_[0] ;
  wire \u_tester/tx_cnt_reg_n_0_[10] ;
  wire \u_tester/tx_cnt_reg_n_0_[11] ;
  wire \u_tester/tx_cnt_reg_n_0_[12] ;
  wire \u_tester/tx_cnt_reg_n_0_[13] ;
  wire \u_tester/tx_cnt_reg_n_0_[14] ;
  wire \u_tester/tx_cnt_reg_n_0_[15] ;
  wire \u_tester/tx_cnt_reg_n_0_[16] ;
  wire \u_tester/tx_cnt_reg_n_0_[17] ;
  wire \u_tester/tx_cnt_reg_n_0_[18] ;
  wire \u_tester/tx_cnt_reg_n_0_[19] ;
  wire \u_tester/tx_cnt_reg_n_0_[1] ;
  wire \u_tester/tx_cnt_reg_n_0_[20] ;
  wire \u_tester/tx_cnt_reg_n_0_[21] ;
  wire \u_tester/tx_cnt_reg_n_0_[22] ;
  wire \u_tester/tx_cnt_reg_n_0_[2] ;
  wire \u_tester/tx_cnt_reg_n_0_[3] ;
  wire \u_tester/tx_cnt_reg_n_0_[4] ;
  wire \u_tester/tx_cnt_reg_n_0_[5] ;
  wire \u_tester/tx_cnt_reg_n_0_[6] ;
  wire \u_tester/tx_cnt_reg_n_0_[7] ;
  wire \u_tester/tx_cnt_reg_n_0_[8] ;
  wire \u_tester/tx_cnt_reg_n_0_[9] ;
  wire \u_tester/u_chk/out_reg[18]_srl3___u_tester_u_chk_out_reg_r_1_n_0 ;
  wire \u_tester/u_chk/out_reg[19]_u_tester_u_chk_out_reg_r_2_n_0 ;
  wire \u_tester/u_chk/out_reg[2]_srl3___u_tester_u_chk_out_reg_r_1_n_0 ;
  wire \u_tester/u_chk/out_reg[34]_srl3___u_tester_u_chk_out_reg_r_1_n_0 ;
  wire \u_tester/u_chk/out_reg[35]_u_tester_u_chk_out_reg_r_2_n_0 ;
  wire \u_tester/u_chk/out_reg[3]_u_tester_u_chk_out_reg_r_2_n_0 ;
  wire \u_tester/u_chk/out_reg[50]_srl3___u_tester_u_chk_out_reg_r_1_n_0 ;
  wire \u_tester/u_chk/out_reg[51]_u_tester_u_chk_out_reg_r_2_n_0 ;
  wire \u_tester/u_chk/out_reg_gate__0_n_0 ;
  wire \u_tester/u_chk/out_reg_gate__1_n_0 ;
  wire \u_tester/u_chk/out_reg_gate__2_n_0 ;
  wire \u_tester/u_chk/out_reg_gate_n_0 ;
  wire \u_tester/u_chk/out_reg_r_0_n_0 ;
  wire \u_tester/u_chk/out_reg_r_1_n_0 ;
  wire \u_tester/u_chk/out_reg_r_2_n_0 ;
  wire \u_tester/u_chk/out_reg_r_n_0 ;
  wire \u_tester/u_gen/out_reg_n_0_[0] ;
  wire \u_tester/u_gen/out_reg_n_0_[10] ;
  wire \u_tester/u_gen/out_reg_n_0_[11] ;
  wire \u_tester/u_gen/out_reg_n_0_[12] ;
  wire \u_tester/u_gen/out_reg_n_0_[13] ;
  wire \u_tester/u_gen/out_reg_n_0_[14] ;
  wire \u_tester/u_gen/out_reg_n_0_[15] ;
  wire \u_tester/u_gen/out_reg_n_0_[16] ;
  wire \u_tester/u_gen/out_reg_n_0_[17] ;
  wire \u_tester/u_gen/out_reg_n_0_[18] ;
  wire \u_tester/u_gen/out_reg_n_0_[19] ;
  wire \u_tester/u_gen/out_reg_n_0_[1] ;
  wire \u_tester/u_gen/out_reg_n_0_[20] ;
  wire \u_tester/u_gen/out_reg_n_0_[21] ;
  wire \u_tester/u_gen/out_reg_n_0_[22] ;
  wire \u_tester/u_gen/out_reg_n_0_[23] ;
  wire \u_tester/u_gen/out_reg_n_0_[24] ;
  wire \u_tester/u_gen/out_reg_n_0_[25] ;
  wire \u_tester/u_gen/out_reg_n_0_[26] ;
  wire \u_tester/u_gen/out_reg_n_0_[27] ;
  wire \u_tester/u_gen/out_reg_n_0_[28] ;
  wire \u_tester/u_gen/out_reg_n_0_[29] ;
  wire \u_tester/u_gen/out_reg_n_0_[2] ;
  wire \u_tester/u_gen/out_reg_n_0_[30] ;
  wire \u_tester/u_gen/out_reg_n_0_[31] ;
  wire \u_tester/u_gen/out_reg_n_0_[32] ;
  wire \u_tester/u_gen/out_reg_n_0_[33] ;
  wire \u_tester/u_gen/out_reg_n_0_[34] ;
  wire \u_tester/u_gen/out_reg_n_0_[35] ;
  wire \u_tester/u_gen/out_reg_n_0_[36] ;
  wire \u_tester/u_gen/out_reg_n_0_[37] ;
  wire \u_tester/u_gen/out_reg_n_0_[38] ;
  wire \u_tester/u_gen/out_reg_n_0_[39] ;
  wire \u_tester/u_gen/out_reg_n_0_[3] ;
  wire \u_tester/u_gen/out_reg_n_0_[40] ;
  wire \u_tester/u_gen/out_reg_n_0_[41] ;
  wire \u_tester/u_gen/out_reg_n_0_[42] ;
  wire \u_tester/u_gen/out_reg_n_0_[43] ;
  wire \u_tester/u_gen/out_reg_n_0_[44] ;
  wire \u_tester/u_gen/out_reg_n_0_[45] ;
  wire \u_tester/u_gen/out_reg_n_0_[46] ;
  wire \u_tester/u_gen/out_reg_n_0_[47] ;
  wire \u_tester/u_gen/out_reg_n_0_[48] ;
  wire \u_tester/u_gen/out_reg_n_0_[49] ;
  wire \u_tester/u_gen/out_reg_n_0_[4] ;
  wire \u_tester/u_gen/out_reg_n_0_[50] ;
  wire \u_tester/u_gen/out_reg_n_0_[51] ;
  wire \u_tester/u_gen/out_reg_n_0_[52] ;
  wire \u_tester/u_gen/out_reg_n_0_[53] ;
  wire \u_tester/u_gen/out_reg_n_0_[54] ;
  wire \u_tester/u_gen/out_reg_n_0_[55] ;
  wire \u_tester/u_gen/out_reg_n_0_[56] ;
  wire \u_tester/u_gen/out_reg_n_0_[57] ;
  wire \u_tester/u_gen/out_reg_n_0_[58] ;
  wire \u_tester/u_gen/out_reg_n_0_[59] ;
  wire \u_tester/u_gen/out_reg_n_0_[5] ;
  wire \u_tester/u_gen/out_reg_n_0_[60] ;
  wire \u_tester/u_gen/out_reg_n_0_[61] ;
  wire \u_tester/u_gen/out_reg_n_0_[62] ;
  wire \u_tester/u_gen/out_reg_n_0_[63] ;
  wire \u_tester/u_gen/out_reg_n_0_[6] ;
  wire \u_tester/u_gen/out_reg_n_0_[7] ;
  wire \u_tester/u_gen/out_reg_n_0_[8] ;
  wire \u_tester/u_gen/out_reg_n_0_[9] ;
  wire [3:3]\NLW_RX_pattern_reg[28]_i_1_CO_UNCONNECTED ;
  wire [3:3]\NLW_counter_src_reg[28]_i_1_CO_UNCONNECTED ;
  wire \NLW_trxiq_rx/iddr_sel_Q2_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[0].oddr_data_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[0].oddr_data_S_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[10].oddr_data_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[10].oddr_data_S_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[11].oddr_data_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[11].oddr_data_S_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[1].oddr_data_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[1].oddr_data_S_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[2].oddr_data_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[2].oddr_data_S_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[3].oddr_data_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[3].oddr_data_S_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[4].oddr_data_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[4].oddr_data_S_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[5].oddr_data_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[5].oddr_data_S_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[6].oddr_data_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[6].oddr_data_S_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[7].oddr_data_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[7].oddr_data_S_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[8].oddr_data_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[8].oddr_data_S_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[9].oddr_data_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/gen_diq_bits[9].oddr_data_S_UNCONNECTED ;
  wire \NLW_trxiq_tx/oddr_sel_R_UNCONNECTED ;
  wire \NLW_trxiq_tx/oddr_sel_S_UNCONNECTED ;
  wire [3:2]\NLW_tx_cnt_reg[28]_i_1_CO_UNCONNECTED ;
  wire [3:3]\NLW_tx_cnt_reg[28]_i_1_O_UNCONNECTED ;
  wire \NLW_u_clocks/mmcm_mclk_CLKOUT3_UNCONNECTED ;
  wire \NLW_u_clocks/mmcm_mclk_CLKOUT4_UNCONNECTED ;
  wire \NLW_u_clocks/mmcm_mclk_CLKOUT5_UNCONNECTED ;
  wire \NLW_u_clocks/mmcm_mclk_LOCKED_UNCONNECTED ;
  wire \NLW_u_clocks/oddr_fclk_R_UNCONNECTED ;
  wire \NLW_u_clocks/oddr_fclk_S_UNCONNECTED ;
  wire [3:0]\NLW_u_tester/rx_err0_carry_O_UNCONNECTED ;
  wire [3:0]\NLW_u_tester/rx_err0_carry__0_O_UNCONNECTED ;
  wire [3:0]\NLW_u_tester/rx_err0_carry__1_O_UNCONNECTED ;
  wire [3:0]\NLW_u_tester/rx_err0_carry__2_O_UNCONNECTED ;
  wire [3:3]\NLW_u_tester/rx_err0_carry__3_CO_UNCONNECTED ;
  wire [3:0]\NLW_u_tester/rx_err0_carry__3_O_UNCONNECTED ;

  assign LMS_DIQ1_TXNRX = \<const1> ;
  assign LMS_DIQ2_FCLK = \<const0> ;
  assign LMS_DIQ2_TXNRX = \<const0> ;
  LUT5 #(
    .INIT(32'h0000EA2A)) 
    EXT_rst_i_1
       (.I0(EXT_rst),
        .I1(SYS_config_data[29]),
        .I2(SYS_config_write),
        .I3(SYS_config_data[0]),
        .I4(SYS_rst),
        .O(EXT_rst_i_1_n_0));
  FDRE EXT_rst_reg
       (.C(SYS_clk),
        .CE(1'b1),
        .D(EXT_rst_i_1_n_0),
        .Q(EXT_rst),
        .R(1'b0));
  GND GND
       (.G(\<const0> ));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT4 #(
    .INIT(16'hAAC0)) 
    \RX_data_a[0]_i_1 
       (.I0(TX_data_a_to_lms[0]),
        .I1(RX_pattern_reg[0]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_a[0]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[10]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[10]),
        .I2(RX_data_a_from_lms[10]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[10]),
        .O(\RX_data_a[10]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[11]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[11]),
        .I2(RX_data_a_from_lms[11]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[11]),
        .O(\RX_data_a[11]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[12]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[12]),
        .I2(RX_data_a_from_lms[12]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[12]),
        .O(\RX_data_a[12]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[13]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[13]),
        .I2(RX_data_a_from_lms[13]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[13]),
        .O(\RX_data_a[13]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[14]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[14]),
        .I2(RX_data_a_from_lms[14]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[14]),
        .O(\RX_data_a[14]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[15]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[15]),
        .I2(RX_data_a_from_lms[15]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[15]),
        .O(\RX_data_a[15]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT4 #(
    .INIT(16'hAAC0)) 
    \RX_data_a[16]_i_1 
       (.I0(TX_data_a_to_lms[16]),
        .I1(RX_pattern_reg[16]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_a[16]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair33" *) 
  LUT4 #(
    .INIT(16'hAAC0)) 
    \RX_data_a[17]_i_1 
       (.I0(TX_data_a_to_lms[17]),
        .I1(RX_pattern_reg[17]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_a[17]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair34" *) 
  LUT4 #(
    .INIT(16'hAAC0)) 
    \RX_data_a[18]_i_1 
       (.I0(TX_data_a_to_lms[18]),
        .I1(RX_pattern_reg[18]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_a[18]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair35" *) 
  LUT4 #(
    .INIT(16'hAAC0)) 
    \RX_data_a[19]_i_1 
       (.I0(TX_data_a_to_lms[19]),
        .I1(RX_pattern_reg[19]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_a[19]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT4 #(
    .INIT(16'hAAC0)) 
    \RX_data_a[1]_i_1 
       (.I0(TX_data_a_to_lms[1]),
        .I1(RX_pattern_reg[1]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_a[1]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[20]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[20]),
        .I2(RX_data_a_from_lms[20]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[20]),
        .O(\RX_data_a[20]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[21]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[21]),
        .I2(RX_data_a_from_lms[21]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[21]),
        .O(\RX_data_a[21]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[22]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[22]),
        .I2(RX_data_a_from_lms[22]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[22]),
        .O(\RX_data_a[22]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[23]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[23]),
        .I2(RX_data_a_from_lms[23]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[23]),
        .O(\RX_data_a[23]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[24]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[24]),
        .I2(RX_data_a_from_lms[24]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[24]),
        .O(\RX_data_a[24]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[25]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[25]),
        .I2(RX_data_a_from_lms[25]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[25]),
        .O(\RX_data_a[25]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[26]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[26]),
        .I2(RX_data_a_from_lms[26]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[26]),
        .O(\RX_data_a[26]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[27]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[27]),
        .I2(RX_data_a_from_lms[27]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[27]),
        .O(\RX_data_a[27]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[28]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[28]),
        .I2(RX_data_a_from_lms[28]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[28]),
        .O(\RX_data_a[28]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[29]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[29]),
        .I2(RX_data_a_from_lms[29]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[29]),
        .O(\RX_data_a[29]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT4 #(
    .INIT(16'hAAC0)) 
    \RX_data_a[2]_i_1 
       (.I0(TX_data_a_to_lms[2]),
        .I1(RX_pattern_reg[2]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_a[2]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[30]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[30]),
        .I2(RX_data_a_from_lms[30]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[30]),
        .O(\RX_data_a[30]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[31]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[31]),
        .I2(RX_data_a_from_lms[31]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[31]),
        .O(\RX_data_a[31]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT4 #(
    .INIT(16'hAAC0)) 
    \RX_data_a[3]_i_1 
       (.I0(TX_data_a_to_lms[3]),
        .I1(RX_pattern_reg[3]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_a[3]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[4]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[4]),
        .I2(RX_data_a_from_lms[4]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[4]),
        .O(\RX_data_a[4]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[5]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[5]),
        .I2(RX_data_a_from_lms[5]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[5]),
        .O(\RX_data_a[5]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[6]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[6]),
        .I2(RX_data_a_from_lms[6]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[6]),
        .O(\RX_data_a[6]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[7]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[7]),
        .I2(RX_data_a_from_lms[7]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[7]),
        .O(\RX_data_a[7]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[8]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[8]),
        .I2(RX_data_a_from_lms[8]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[8]),
        .O(\RX_data_a[8]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFD800D8)) 
    \RX_data_a[9]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[9]),
        .I2(RX_data_a_from_lms[9]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_a_to_lms[9]),
        .O(\RX_data_a[9]_i_1_n_0 ));
  FDRE \RX_data_a_reg[0] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[0]_i_1_n_0 ),
        .Q(RX_data_a[0]),
        .R(1'b0));
  FDRE \RX_data_a_reg[10] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[10]_i_1_n_0 ),
        .Q(RX_data_a[10]),
        .R(1'b0));
  FDRE \RX_data_a_reg[11] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[11]_i_1_n_0 ),
        .Q(RX_data_a[11]),
        .R(1'b0));
  FDRE \RX_data_a_reg[12] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[12]_i_1_n_0 ),
        .Q(RX_data_a[12]),
        .R(1'b0));
  FDRE \RX_data_a_reg[13] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[13]_i_1_n_0 ),
        .Q(RX_data_a[13]),
        .R(1'b0));
  FDRE \RX_data_a_reg[14] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[14]_i_1_n_0 ),
        .Q(RX_data_a[14]),
        .R(1'b0));
  FDRE \RX_data_a_reg[15] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[15]_i_1_n_0 ),
        .Q(RX_data_a[15]),
        .R(1'b0));
  FDRE \RX_data_a_reg[16] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[16]_i_1_n_0 ),
        .Q(RX_data_a[16]),
        .R(1'b0));
  FDRE \RX_data_a_reg[17] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[17]_i_1_n_0 ),
        .Q(RX_data_a[17]),
        .R(1'b0));
  FDRE \RX_data_a_reg[18] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[18]_i_1_n_0 ),
        .Q(RX_data_a[18]),
        .R(1'b0));
  FDRE \RX_data_a_reg[19] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[19]_i_1_n_0 ),
        .Q(RX_data_a[19]),
        .R(1'b0));
  FDRE \RX_data_a_reg[1] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[1]_i_1_n_0 ),
        .Q(RX_data_a[1]),
        .R(1'b0));
  FDRE \RX_data_a_reg[20] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[20]_i_1_n_0 ),
        .Q(RX_data_a[20]),
        .R(1'b0));
  FDRE \RX_data_a_reg[21] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[21]_i_1_n_0 ),
        .Q(RX_data_a[21]),
        .R(1'b0));
  FDRE \RX_data_a_reg[22] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[22]_i_1_n_0 ),
        .Q(RX_data_a[22]),
        .R(1'b0));
  FDRE \RX_data_a_reg[23] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[23]_i_1_n_0 ),
        .Q(RX_data_a[23]),
        .R(1'b0));
  FDRE \RX_data_a_reg[24] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[24]_i_1_n_0 ),
        .Q(RX_data_a[24]),
        .R(1'b0));
  FDRE \RX_data_a_reg[25] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[25]_i_1_n_0 ),
        .Q(RX_data_a[25]),
        .R(1'b0));
  FDRE \RX_data_a_reg[26] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[26]_i_1_n_0 ),
        .Q(RX_data_a[26]),
        .R(1'b0));
  FDRE \RX_data_a_reg[27] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[27]_i_1_n_0 ),
        .Q(RX_data_a[27]),
        .R(1'b0));
  FDRE \RX_data_a_reg[28] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[28]_i_1_n_0 ),
        .Q(RX_data_a[28]),
        .R(1'b0));
  FDRE \RX_data_a_reg[29] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[29]_i_1_n_0 ),
        .Q(RX_data_a[29]),
        .R(1'b0));
  FDRE \RX_data_a_reg[2] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[2]_i_1_n_0 ),
        .Q(RX_data_a[2]),
        .R(1'b0));
  FDRE \RX_data_a_reg[30] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[30]_i_1_n_0 ),
        .Q(RX_data_a[30]),
        .R(1'b0));
  FDRE \RX_data_a_reg[31] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[31]_i_1_n_0 ),
        .Q(RX_data_a[31]),
        .R(1'b0));
  FDRE \RX_data_a_reg[3] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[3]_i_1_n_0 ),
        .Q(RX_data_a[3]),
        .R(1'b0));
  FDRE \RX_data_a_reg[4] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[4]_i_1_n_0 ),
        .Q(RX_data_a[4]),
        .R(1'b0));
  FDRE \RX_data_a_reg[5] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[5]_i_1_n_0 ),
        .Q(RX_data_a[5]),
        .R(1'b0));
  FDRE \RX_data_a_reg[6] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[6]_i_1_n_0 ),
        .Q(RX_data_a[6]),
        .R(1'b0));
  FDRE \RX_data_a_reg[7] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[7]_i_1_n_0 ),
        .Q(RX_data_a[7]),
        .R(1'b0));
  FDRE \RX_data_a_reg[8] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[8]_i_1_n_0 ),
        .Q(RX_data_a[8]),
        .R(1'b0));
  FDRE \RX_data_a_reg[9] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_a[9]_i_1_n_0 ),
        .Q(RX_data_a[9]),
        .R(1'b0));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT4 #(
    .INIT(16'hAA30)) 
    \RX_data_b[0]_i_1 
       (.I0(TX_data_b_to_lms[0]),
        .I1(RX_pattern_reg[0]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_b[0]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[10]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[10]),
        .I2(RX_data_b_from_lms[10]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[10]),
        .O(\RX_data_b[10]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[11]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[11]),
        .I2(RX_data_b_from_lms[11]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[11]),
        .O(\RX_data_b[11]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[12]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[12]),
        .I2(RX_data_b_from_lms[12]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[12]),
        .O(\RX_data_b[12]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[13]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[13]),
        .I2(RX_data_b_from_lms[13]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[13]),
        .O(\RX_data_b[13]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[14]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[14]),
        .I2(RX_data_b_from_lms[14]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[14]),
        .O(\RX_data_b[14]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[15]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[15]),
        .I2(RX_data_b_from_lms[15]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[15]),
        .O(\RX_data_b[15]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT4 #(
    .INIT(16'hAA30)) 
    \RX_data_b[16]_i_1 
       (.I0(TX_data_b_to_lms[16]),
        .I1(RX_pattern_reg[16]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_b[16]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair33" *) 
  LUT4 #(
    .INIT(16'hAA30)) 
    \RX_data_b[17]_i_1 
       (.I0(TX_data_b_to_lms[17]),
        .I1(RX_pattern_reg[17]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_b[17]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair34" *) 
  LUT4 #(
    .INIT(16'hAA30)) 
    \RX_data_b[18]_i_1 
       (.I0(TX_data_b_to_lms[18]),
        .I1(RX_pattern_reg[18]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_b[18]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair35" *) 
  LUT4 #(
    .INIT(16'hAA30)) 
    \RX_data_b[19]_i_1 
       (.I0(TX_data_b_to_lms[19]),
        .I1(RX_pattern_reg[19]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_b[19]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT4 #(
    .INIT(16'hAA30)) 
    \RX_data_b[1]_i_1 
       (.I0(TX_data_b_to_lms[1]),
        .I1(RX_pattern_reg[1]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_b[1]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[20]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[20]),
        .I2(RX_data_b_from_lms[20]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[20]),
        .O(\RX_data_b[20]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[21]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[21]),
        .I2(RX_data_b_from_lms[21]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[21]),
        .O(\RX_data_b[21]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[22]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[22]),
        .I2(RX_data_b_from_lms[22]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[22]),
        .O(\RX_data_b[22]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[23]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[23]),
        .I2(RX_data_b_from_lms[23]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[23]),
        .O(\RX_data_b[23]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[24]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[24]),
        .I2(RX_data_b_from_lms[24]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[24]),
        .O(\RX_data_b[24]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[25]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[25]),
        .I2(RX_data_b_from_lms[25]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[25]),
        .O(\RX_data_b[25]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[26]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[26]),
        .I2(RX_data_b_from_lms[26]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[26]),
        .O(\RX_data_b[26]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[27]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[27]),
        .I2(RX_data_b_from_lms[27]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[27]),
        .O(\RX_data_b[27]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[28]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[28]),
        .I2(RX_data_b_from_lms[28]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[28]),
        .O(\RX_data_b[28]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[29]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[29]),
        .I2(RX_data_b_from_lms[29]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[29]),
        .O(\RX_data_b[29]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT4 #(
    .INIT(16'hAA30)) 
    \RX_data_b[2]_i_1 
       (.I0(TX_data_b_to_lms[2]),
        .I1(RX_pattern_reg[2]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_b[2]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[30]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[30]),
        .I2(RX_data_b_from_lms[30]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[30]),
        .O(\RX_data_b[30]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[31]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[31]),
        .I2(RX_data_b_from_lms[31]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[31]),
        .O(\RX_data_b[31]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT4 #(
    .INIT(16'hAA30)) 
    \RX_data_b[3]_i_1 
       (.I0(TX_data_b_to_lms[3]),
        .I1(RX_pattern_reg[3]),
        .I2(rx_test_mode_xfer_n_1),
        .I3(rx_test_mode_xfer_n_0),
        .O(\RX_data_b[3]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[4]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[4]),
        .I2(RX_data_b_from_lms[4]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[4]),
        .O(\RX_data_b[4]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[5]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[5]),
        .I2(RX_data_b_from_lms[5]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[5]),
        .O(\RX_data_b[5]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[6]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[6]),
        .I2(RX_data_b_from_lms[6]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[6]),
        .O(\RX_data_b[6]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[7]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[7]),
        .I2(RX_data_b_from_lms[7]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[7]),
        .O(\RX_data_b[7]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[8]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[8]),
        .I2(RX_data_b_from_lms[8]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[8]),
        .O(\RX_data_b[8]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF720072)) 
    \RX_data_b[9]_i_1 
       (.I0(rx_test_mode_xfer_n_1),
        .I1(RX_pattern_reg[9]),
        .I2(RX_data_b_from_lms[9]),
        .I3(rx_test_mode_xfer_n_0),
        .I4(TX_data_b_to_lms[9]),
        .O(\RX_data_b[9]_i_1_n_0 ));
  FDRE \RX_data_b_reg[0] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[0]_i_1_n_0 ),
        .Q(RX_data_b[0]),
        .R(1'b0));
  FDRE \RX_data_b_reg[10] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[10]_i_1_n_0 ),
        .Q(RX_data_b[10]),
        .R(1'b0));
  FDRE \RX_data_b_reg[11] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[11]_i_1_n_0 ),
        .Q(RX_data_b[11]),
        .R(1'b0));
  FDRE \RX_data_b_reg[12] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[12]_i_1_n_0 ),
        .Q(RX_data_b[12]),
        .R(1'b0));
  FDRE \RX_data_b_reg[13] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[13]_i_1_n_0 ),
        .Q(RX_data_b[13]),
        .R(1'b0));
  FDRE \RX_data_b_reg[14] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[14]_i_1_n_0 ),
        .Q(RX_data_b[14]),
        .R(1'b0));
  FDRE \RX_data_b_reg[15] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[15]_i_1_n_0 ),
        .Q(RX_data_b[15]),
        .R(1'b0));
  FDRE \RX_data_b_reg[16] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[16]_i_1_n_0 ),
        .Q(RX_data_b[16]),
        .R(1'b0));
  FDRE \RX_data_b_reg[17] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[17]_i_1_n_0 ),
        .Q(RX_data_b[17]),
        .R(1'b0));
  FDRE \RX_data_b_reg[18] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[18]_i_1_n_0 ),
        .Q(RX_data_b[18]),
        .R(1'b0));
  FDRE \RX_data_b_reg[19] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[19]_i_1_n_0 ),
        .Q(RX_data_b[19]),
        .R(1'b0));
  FDRE \RX_data_b_reg[1] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[1]_i_1_n_0 ),
        .Q(RX_data_b[1]),
        .R(1'b0));
  FDRE \RX_data_b_reg[20] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[20]_i_1_n_0 ),
        .Q(RX_data_b[20]),
        .R(1'b0));
  FDRE \RX_data_b_reg[21] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[21]_i_1_n_0 ),
        .Q(RX_data_b[21]),
        .R(1'b0));
  FDRE \RX_data_b_reg[22] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[22]_i_1_n_0 ),
        .Q(RX_data_b[22]),
        .R(1'b0));
  FDRE \RX_data_b_reg[23] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[23]_i_1_n_0 ),
        .Q(RX_data_b[23]),
        .R(1'b0));
  FDRE \RX_data_b_reg[24] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[24]_i_1_n_0 ),
        .Q(RX_data_b[24]),
        .R(1'b0));
  FDRE \RX_data_b_reg[25] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[25]_i_1_n_0 ),
        .Q(RX_data_b[25]),
        .R(1'b0));
  FDRE \RX_data_b_reg[26] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[26]_i_1_n_0 ),
        .Q(RX_data_b[26]),
        .R(1'b0));
  FDRE \RX_data_b_reg[27] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[27]_i_1_n_0 ),
        .Q(RX_data_b[27]),
        .R(1'b0));
  FDRE \RX_data_b_reg[28] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[28]_i_1_n_0 ),
        .Q(RX_data_b[28]),
        .R(1'b0));
  FDRE \RX_data_b_reg[29] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[29]_i_1_n_0 ),
        .Q(RX_data_b[29]),
        .R(1'b0));
  FDRE \RX_data_b_reg[2] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[2]_i_1_n_0 ),
        .Q(RX_data_b[2]),
        .R(1'b0));
  FDRE \RX_data_b_reg[30] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[30]_i_1_n_0 ),
        .Q(RX_data_b[30]),
        .R(1'b0));
  FDRE \RX_data_b_reg[31] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[31]_i_1_n_0 ),
        .Q(RX_data_b[31]),
        .R(1'b0));
  FDRE \RX_data_b_reg[3] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[3]_i_1_n_0 ),
        .Q(RX_data_b[3]),
        .R(1'b0));
  FDRE \RX_data_b_reg[4] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[4]_i_1_n_0 ),
        .Q(RX_data_b[4]),
        .R(1'b0));
  FDRE \RX_data_b_reg[5] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[5]_i_1_n_0 ),
        .Q(RX_data_b[5]),
        .R(1'b0));
  FDRE \RX_data_b_reg[6] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[6]_i_1_n_0 ),
        .Q(RX_data_b[6]),
        .R(1'b0));
  FDRE \RX_data_b_reg[7] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[7]_i_1_n_0 ),
        .Q(RX_data_b[7]),
        .R(1'b0));
  FDRE \RX_data_b_reg[8] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[8]_i_1_n_0 ),
        .Q(RX_data_b[8]),
        .R(1'b0));
  FDRE \RX_data_b_reg[9] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\RX_data_b[9]_i_1_n_0 ),
        .Q(RX_data_b[9]),
        .R(1'b0));
  LUT1 #(
    .INIT(2'h1)) 
    \RX_pattern[0]_i_2 
       (.I0(RX_pattern_reg[0]),
        .O(RX_data_b0));
  FDRE \RX_pattern_reg[0] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[0]_i_1_n_7 ),
        .Q(RX_pattern_reg[0]),
        .R(1'b0));
  CARRY4 \RX_pattern_reg[0]_i_1 
       (.CI(1'b0),
        .CO({\RX_pattern_reg[0]_i_1_n_0 ,\RX_pattern_reg[0]_i_1_n_1 ,\RX_pattern_reg[0]_i_1_n_2 ,\RX_pattern_reg[0]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b1}),
        .O({\RX_pattern_reg[0]_i_1_n_4 ,\RX_pattern_reg[0]_i_1_n_5 ,\RX_pattern_reg[0]_i_1_n_6 ,\RX_pattern_reg[0]_i_1_n_7 }),
        .S({RX_pattern_reg[3:1],RX_data_b0}));
  FDRE \RX_pattern_reg[10] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[8]_i_1_n_5 ),
        .Q(RX_pattern_reg[10]),
        .R(1'b0));
  FDRE \RX_pattern_reg[11] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[8]_i_1_n_4 ),
        .Q(RX_pattern_reg[11]),
        .R(1'b0));
  FDRE \RX_pattern_reg[12] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[12]_i_1_n_7 ),
        .Q(RX_pattern_reg[12]),
        .R(1'b0));
  CARRY4 \RX_pattern_reg[12]_i_1 
       (.CI(\RX_pattern_reg[8]_i_1_n_0 ),
        .CO({\RX_pattern_reg[12]_i_1_n_0 ,\RX_pattern_reg[12]_i_1_n_1 ,\RX_pattern_reg[12]_i_1_n_2 ,\RX_pattern_reg[12]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\RX_pattern_reg[12]_i_1_n_4 ,\RX_pattern_reg[12]_i_1_n_5 ,\RX_pattern_reg[12]_i_1_n_6 ,\RX_pattern_reg[12]_i_1_n_7 }),
        .S(RX_pattern_reg[15:12]));
  FDRE \RX_pattern_reg[13] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[12]_i_1_n_6 ),
        .Q(RX_pattern_reg[13]),
        .R(1'b0));
  FDRE \RX_pattern_reg[14] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[12]_i_1_n_5 ),
        .Q(RX_pattern_reg[14]),
        .R(1'b0));
  FDRE \RX_pattern_reg[15] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[12]_i_1_n_4 ),
        .Q(RX_pattern_reg[15]),
        .R(1'b0));
  FDRE \RX_pattern_reg[16] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[16]_i_1_n_7 ),
        .Q(RX_pattern_reg[16]),
        .R(1'b0));
  CARRY4 \RX_pattern_reg[16]_i_1 
       (.CI(\RX_pattern_reg[12]_i_1_n_0 ),
        .CO({\RX_pattern_reg[16]_i_1_n_0 ,\RX_pattern_reg[16]_i_1_n_1 ,\RX_pattern_reg[16]_i_1_n_2 ,\RX_pattern_reg[16]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\RX_pattern_reg[16]_i_1_n_4 ,\RX_pattern_reg[16]_i_1_n_5 ,\RX_pattern_reg[16]_i_1_n_6 ,\RX_pattern_reg[16]_i_1_n_7 }),
        .S(RX_pattern_reg[19:16]));
  FDRE \RX_pattern_reg[17] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[16]_i_1_n_6 ),
        .Q(RX_pattern_reg[17]),
        .R(1'b0));
  FDRE \RX_pattern_reg[18] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[16]_i_1_n_5 ),
        .Q(RX_pattern_reg[18]),
        .R(1'b0));
  FDRE \RX_pattern_reg[19] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[16]_i_1_n_4 ),
        .Q(RX_pattern_reg[19]),
        .R(1'b0));
  FDRE \RX_pattern_reg[1] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[0]_i_1_n_6 ),
        .Q(RX_pattern_reg[1]),
        .R(1'b0));
  FDRE \RX_pattern_reg[20] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[20]_i_1_n_7 ),
        .Q(RX_pattern_reg[20]),
        .R(1'b0));
  CARRY4 \RX_pattern_reg[20]_i_1 
       (.CI(\RX_pattern_reg[16]_i_1_n_0 ),
        .CO({\RX_pattern_reg[20]_i_1_n_0 ,\RX_pattern_reg[20]_i_1_n_1 ,\RX_pattern_reg[20]_i_1_n_2 ,\RX_pattern_reg[20]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\RX_pattern_reg[20]_i_1_n_4 ,\RX_pattern_reg[20]_i_1_n_5 ,\RX_pattern_reg[20]_i_1_n_6 ,\RX_pattern_reg[20]_i_1_n_7 }),
        .S(RX_pattern_reg[23:20]));
  FDRE \RX_pattern_reg[21] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[20]_i_1_n_6 ),
        .Q(RX_pattern_reg[21]),
        .R(1'b0));
  FDRE \RX_pattern_reg[22] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[20]_i_1_n_5 ),
        .Q(RX_pattern_reg[22]),
        .R(1'b0));
  FDRE \RX_pattern_reg[23] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[20]_i_1_n_4 ),
        .Q(RX_pattern_reg[23]),
        .R(1'b0));
  FDRE \RX_pattern_reg[24] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[24]_i_1_n_7 ),
        .Q(RX_pattern_reg[24]),
        .R(1'b0));
  CARRY4 \RX_pattern_reg[24]_i_1 
       (.CI(\RX_pattern_reg[20]_i_1_n_0 ),
        .CO({\RX_pattern_reg[24]_i_1_n_0 ,\RX_pattern_reg[24]_i_1_n_1 ,\RX_pattern_reg[24]_i_1_n_2 ,\RX_pattern_reg[24]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\RX_pattern_reg[24]_i_1_n_4 ,\RX_pattern_reg[24]_i_1_n_5 ,\RX_pattern_reg[24]_i_1_n_6 ,\RX_pattern_reg[24]_i_1_n_7 }),
        .S(RX_pattern_reg[27:24]));
  FDRE \RX_pattern_reg[25] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[24]_i_1_n_6 ),
        .Q(RX_pattern_reg[25]),
        .R(1'b0));
  FDRE \RX_pattern_reg[26] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[24]_i_1_n_5 ),
        .Q(RX_pattern_reg[26]),
        .R(1'b0));
  FDRE \RX_pattern_reg[27] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[24]_i_1_n_4 ),
        .Q(RX_pattern_reg[27]),
        .R(1'b0));
  FDRE \RX_pattern_reg[28] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[28]_i_1_n_7 ),
        .Q(RX_pattern_reg[28]),
        .R(1'b0));
  CARRY4 \RX_pattern_reg[28]_i_1 
       (.CI(\RX_pattern_reg[24]_i_1_n_0 ),
        .CO({\NLW_RX_pattern_reg[28]_i_1_CO_UNCONNECTED [3],\RX_pattern_reg[28]_i_1_n_1 ,\RX_pattern_reg[28]_i_1_n_2 ,\RX_pattern_reg[28]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\RX_pattern_reg[28]_i_1_n_4 ,\RX_pattern_reg[28]_i_1_n_5 ,\RX_pattern_reg[28]_i_1_n_6 ,\RX_pattern_reg[28]_i_1_n_7 }),
        .S(RX_pattern_reg[31:28]));
  FDRE \RX_pattern_reg[29] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[28]_i_1_n_6 ),
        .Q(RX_pattern_reg[29]),
        .R(1'b0));
  FDRE \RX_pattern_reg[2] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[0]_i_1_n_5 ),
        .Q(RX_pattern_reg[2]),
        .R(1'b0));
  FDRE \RX_pattern_reg[30] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[28]_i_1_n_5 ),
        .Q(RX_pattern_reg[30]),
        .R(1'b0));
  FDRE \RX_pattern_reg[31] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[28]_i_1_n_4 ),
        .Q(RX_pattern_reg[31]),
        .R(1'b0));
  FDRE \RX_pattern_reg[3] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[0]_i_1_n_4 ),
        .Q(RX_pattern_reg[3]),
        .R(1'b0));
  FDRE \RX_pattern_reg[4] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[4]_i_1_n_7 ),
        .Q(RX_pattern_reg[4]),
        .R(1'b0));
  CARRY4 \RX_pattern_reg[4]_i_1 
       (.CI(\RX_pattern_reg[0]_i_1_n_0 ),
        .CO({\RX_pattern_reg[4]_i_1_n_0 ,\RX_pattern_reg[4]_i_1_n_1 ,\RX_pattern_reg[4]_i_1_n_2 ,\RX_pattern_reg[4]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\RX_pattern_reg[4]_i_1_n_4 ,\RX_pattern_reg[4]_i_1_n_5 ,\RX_pattern_reg[4]_i_1_n_6 ,\RX_pattern_reg[4]_i_1_n_7 }),
        .S(RX_pattern_reg[7:4]));
  FDRE \RX_pattern_reg[5] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[4]_i_1_n_6 ),
        .Q(RX_pattern_reg[5]),
        .R(1'b0));
  FDRE \RX_pattern_reg[6] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[4]_i_1_n_5 ),
        .Q(RX_pattern_reg[6]),
        .R(1'b0));
  FDRE \RX_pattern_reg[7] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[4]_i_1_n_4 ),
        .Q(RX_pattern_reg[7]),
        .R(1'b0));
  FDRE \RX_pattern_reg[8] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[8]_i_1_n_7 ),
        .Q(RX_pattern_reg[8]),
        .R(1'b0));
  CARRY4 \RX_pattern_reg[8]_i_1 
       (.CI(\RX_pattern_reg[4]_i_1_n_0 ),
        .CO({\RX_pattern_reg[8]_i_1_n_0 ,\RX_pattern_reg[8]_i_1_n_1 ,\RX_pattern_reg[8]_i_1_n_2 ,\RX_pattern_reg[8]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\RX_pattern_reg[8]_i_1_n_4 ,\RX_pattern_reg[8]_i_1_n_5 ,\RX_pattern_reg[8]_i_1_n_6 ,\RX_pattern_reg[8]_i_1_n_7 }),
        .S(RX_pattern_reg[11:8]));
  FDRE \RX_pattern_reg[9] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\RX_pattern_reg[8]_i_1_n_6 ),
        .Q(RX_pattern_reg[9]),
        .R(1'b0));
  LUT2 #(
    .INIT(4'h8)) 
    \SYS_test_mode[4]_i_1 
       (.I0(SYS_config_write),
        .I1(SYS_config_data[28]),
        .O(SYS_test_mode0));
  FDRE \SYS_test_mode_reg[0] 
       (.C(SYS_clk),
        .CE(SYS_test_mode0),
        .D(SYS_config_data[0]),
        .Q(SYS_test_mode[0]),
        .R(SYS_rst));
  FDRE \SYS_test_mode_reg[1] 
       (.C(SYS_clk),
        .CE(SYS_test_mode0),
        .D(SYS_config_data[1]),
        .Q(SYS_test_mode[1]),
        .R(SYS_rst));
  FDRE \SYS_test_mode_reg[2] 
       (.C(SYS_clk),
        .CE(SYS_test_mode0),
        .D(SYS_config_data[2]),
        .Q(SYS_test_mode[2]),
        .R(SYS_rst));
  FDRE \SYS_test_mode_reg[3] 
       (.C(SYS_clk),
        .CE(SYS_test_mode0),
        .D(SYS_config_data[3]),
        .Q(SYS_test_mode[3]),
        .R(SYS_rst));
  FDRE \SYS_test_mode_reg[4] 
       (.C(SYS_clk),
        .CE(SYS_test_mode0),
        .D(SYS_config_data[4]),
        .Q(SYS_test_mode[4]),
        .R(SYS_rst));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_a_to_lms[0]_i_1 
       (.I0(tx_test_data_xfer_n_63),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_a_to_lms[0]_i_2_n_0 ),
        .O(\TX_data_a_to_lms[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[0]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[0] ),
        .I2(TX_data_a[0]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[0]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[10]_i_1 
       (.I0(\TX_data_a_to_lms[10]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_53),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[10]),
        .O(\TX_data_a_to_lms[10]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[10]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[10] ),
        .I2(TX_data_a[10]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[10]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[11]_i_1 
       (.I0(\TX_data_a_to_lms[11]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_52),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[11]),
        .O(\TX_data_a_to_lms[11]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[11]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[11] ),
        .I2(TX_data_a[11]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[11]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[12]_i_1 
       (.I0(\TX_data_a_to_lms[12]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_51),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[12]),
        .O(\TX_data_a_to_lms[12]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[12]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[12] ),
        .I2(TX_data_a[12]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[12]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[13]_i_1 
       (.I0(\TX_data_a_to_lms[13]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_50),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[13]),
        .O(\TX_data_a_to_lms[13]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[13]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[13] ),
        .I2(TX_data_a[13]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[13]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[14]_i_1 
       (.I0(\TX_data_a_to_lms[14]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_49),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[14]),
        .O(\TX_data_a_to_lms[14]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[14]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[14] ),
        .I2(TX_data_a[14]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[14]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[15]_i_1 
       (.I0(\TX_data_a_to_lms[15]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_48),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[15]),
        .O(\TX_data_a_to_lms[15]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[15]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[15] ),
        .I2(TX_data_a[15]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[15]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_a_to_lms[16]_i_1 
       (.I0(tx_test_data_xfer_n_47),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_a_to_lms[16]_i_2_n_0 ),
        .O(\TX_data_a_to_lms[16]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[16]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[16] ),
        .I2(TX_data_a[16]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[16]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_a_to_lms[17]_i_1 
       (.I0(tx_test_data_xfer_n_46),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_a_to_lms[17]_i_2_n_0 ),
        .O(\TX_data_a_to_lms[17]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[17]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[17] ),
        .I2(TX_data_a[17]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[17]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_a_to_lms[18]_i_1 
       (.I0(tx_test_data_xfer_n_45),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_a_to_lms[18]_i_2_n_0 ),
        .O(\TX_data_a_to_lms[18]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[18]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[18] ),
        .I2(TX_data_a[18]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[18]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_a_to_lms[19]_i_1 
       (.I0(tx_test_data_xfer_n_44),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_a_to_lms[19]_i_2_n_0 ),
        .O(\TX_data_a_to_lms[19]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[19]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[19] ),
        .I2(TX_data_a[19]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[19]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_a_to_lms[1]_i_1 
       (.I0(tx_test_data_xfer_n_62),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_a_to_lms[1]_i_2_n_0 ),
        .O(\TX_data_a_to_lms[1]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[1]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[1] ),
        .I2(TX_data_a[1]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[1]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[20]_i_1 
       (.I0(\TX_data_a_to_lms[20]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_43),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[20]),
        .O(\TX_data_a_to_lms[20]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[20]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[20] ),
        .I2(TX_data_a[20]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[20]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[21]_i_1 
       (.I0(\TX_data_a_to_lms[21]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_42),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[21]),
        .O(\TX_data_a_to_lms[21]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[21]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[21] ),
        .I2(TX_data_a[21]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[21]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[22]_i_1 
       (.I0(\TX_data_a_to_lms[22]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_41),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[22]),
        .O(\TX_data_a_to_lms[22]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[22]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[22] ),
        .I2(TX_data_a[22]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[22]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[23]_i_1 
       (.I0(\TX_data_a_to_lms[23]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_40),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[23]),
        .O(\TX_data_a_to_lms[23]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[23]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[23] ),
        .I2(TX_data_a[23]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[23]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[24]_i_1 
       (.I0(\TX_data_a_to_lms[24]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_39),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[24]),
        .O(\TX_data_a_to_lms[24]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[24]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[24] ),
        .I2(TX_data_a[24]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[24]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[25]_i_1 
       (.I0(\TX_data_a_to_lms[25]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_38),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[25]),
        .O(\TX_data_a_to_lms[25]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[25]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[25] ),
        .I2(TX_data_a[25]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[25]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[26]_i_1 
       (.I0(\TX_data_a_to_lms[26]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_37),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[26]),
        .O(\TX_data_a_to_lms[26]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[26]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[26] ),
        .I2(TX_data_a[26]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[26]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[27]_i_1 
       (.I0(\TX_data_a_to_lms[27]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_36),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[27]),
        .O(\TX_data_a_to_lms[27]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[27]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[27] ),
        .I2(TX_data_a[27]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[27]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[28]_i_1 
       (.I0(\TX_data_a_to_lms[28]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_35),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[28]),
        .O(\TX_data_a_to_lms[28]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[28]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[28] ),
        .I2(TX_data_a[28]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[28]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[29]_i_1 
       (.I0(\TX_data_a_to_lms[29]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_34),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[29]),
        .O(\TX_data_a_to_lms[29]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[29]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[29] ),
        .I2(TX_data_a[29]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[29]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_a_to_lms[2]_i_1 
       (.I0(tx_test_data_xfer_n_61),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_a_to_lms[2]_i_2_n_0 ),
        .O(\TX_data_a_to_lms[2]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[2]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[2] ),
        .I2(TX_data_a[2]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[2]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[30]_i_1 
       (.I0(\TX_data_a_to_lms[30]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_33),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[30]),
        .O(\TX_data_a_to_lms[30]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[30]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[30] ),
        .I2(TX_data_a[30]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[30]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[31]_i_1 
       (.I0(\TX_data_a_to_lms[31]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_32),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[31]),
        .O(\TX_data_a_to_lms[31]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[31]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[31] ),
        .I2(TX_data_a[31]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[31]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_a_to_lms[3]_i_1 
       (.I0(tx_test_data_xfer_n_60),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_a_to_lms[3]_i_2_n_0 ),
        .O(\TX_data_a_to_lms[3]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[3]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[3] ),
        .I2(TX_data_a[3]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[3]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[4]_i_1 
       (.I0(\TX_data_a_to_lms[4]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_59),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[4]),
        .O(\TX_data_a_to_lms[4]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[4]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[4] ),
        .I2(TX_data_a[4]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[4]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[5]_i_1 
       (.I0(\TX_data_a_to_lms[5]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_58),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[5]),
        .O(\TX_data_a_to_lms[5]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[5]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[5] ),
        .I2(TX_data_a[5]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[5]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[6]_i_1 
       (.I0(\TX_data_a_to_lms[6]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_57),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[6]),
        .O(\TX_data_a_to_lms[6]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[6]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[6] ),
        .I2(TX_data_a[6]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[6]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[7]_i_1 
       (.I0(\TX_data_a_to_lms[7]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_56),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[7]),
        .O(\TX_data_a_to_lms[7]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[7]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[7] ),
        .I2(TX_data_a[7]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[7]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[8]_i_1 
       (.I0(\TX_data_a_to_lms[8]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_55),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[8]),
        .O(\TX_data_a_to_lms[8]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[8]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[8] ),
        .I2(TX_data_a[8]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[8]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_a_to_lms[9]_i_1 
       (.I0(\TX_data_a_to_lms[9]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_data_xfer_n_54),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_a_from_lms[9]),
        .O(\TX_data_a_to_lms[9]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_a_to_lms[9]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[9] ),
        .I2(TX_data_a[9]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_a_to_lms[9]_i_2_n_0 ));
  FDRE \TX_data_a_to_lms_reg[0] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[0]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[0]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[10] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[10]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[10]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[11] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[11]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[11]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[12] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[12]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[12]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[13] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[13]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[13]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[14] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[14]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[14]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[15] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[15]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[15]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[16] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[16]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[16]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[17] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[17]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[17]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[18] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[18]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[18]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[19] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[19]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[19]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[1] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[1]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[1]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[20] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[20]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[20]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[21] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[21]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[21]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[22] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[22]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[22]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[23] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[23]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[23]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[24] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[24]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[24]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[25] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[25]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[25]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[26] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[26]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[26]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[27] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[27]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[27]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[28] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[28]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[28]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[29] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[29]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[29]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[2] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[2]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[2]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[30] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[30]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[30]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[31] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[31]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[31]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[3] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[3]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[3]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[4] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[4]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[4]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[5] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[5]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[5]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[6] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[6]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[6]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[7] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[7]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[7]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[8] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[8]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[8]),
        .R(1'b0));
  FDRE \TX_data_a_to_lms_reg[9] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_a_to_lms[9]_i_1_n_0 ),
        .Q(TX_data_a_to_lms[9]),
        .R(1'b0));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_b_to_lms[0]_i_1 
       (.I0(TX_test_data_b[0]),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_b_to_lms[0]_i_2_n_0 ),
        .O(\TX_data_b_to_lms[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[0]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[32] ),
        .I2(TX_data_b[0]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[0]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[10]_i_1 
       (.I0(\TX_data_b_to_lms[10]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[10]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[10]),
        .O(\TX_data_b_to_lms[10]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[10]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[42] ),
        .I2(TX_data_b[10]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[10]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[11]_i_1 
       (.I0(\TX_data_b_to_lms[11]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[11]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[11]),
        .O(\TX_data_b_to_lms[11]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[11]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[43] ),
        .I2(TX_data_b[11]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[11]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[12]_i_1 
       (.I0(\TX_data_b_to_lms[12]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[12]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[12]),
        .O(\TX_data_b_to_lms[12]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[12]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[44] ),
        .I2(TX_data_b[12]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[12]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[13]_i_1 
       (.I0(\TX_data_b_to_lms[13]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[13]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[13]),
        .O(\TX_data_b_to_lms[13]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[13]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[45] ),
        .I2(TX_data_b[13]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[13]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[14]_i_1 
       (.I0(\TX_data_b_to_lms[14]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[14]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[14]),
        .O(\TX_data_b_to_lms[14]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[14]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[46] ),
        .I2(TX_data_b[14]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[14]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[15]_i_1 
       (.I0(\TX_data_b_to_lms[15]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[15]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[15]),
        .O(\TX_data_b_to_lms[15]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[15]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[47] ),
        .I2(TX_data_b[15]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[15]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_b_to_lms[16]_i_1 
       (.I0(TX_test_data_b[16]),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_b_to_lms[16]_i_2_n_0 ),
        .O(\TX_data_b_to_lms[16]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[16]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[48] ),
        .I2(TX_data_b[16]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[16]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_b_to_lms[17]_i_1 
       (.I0(TX_test_data_b[17]),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_b_to_lms[17]_i_2_n_0 ),
        .O(\TX_data_b_to_lms[17]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[17]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[49] ),
        .I2(TX_data_b[17]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[17]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_b_to_lms[18]_i_1 
       (.I0(TX_test_data_b[18]),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_b_to_lms[18]_i_2_n_0 ),
        .O(\TX_data_b_to_lms[18]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[18]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[50] ),
        .I2(TX_data_b[18]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[18]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_b_to_lms[19]_i_1 
       (.I0(TX_test_data_b[19]),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_b_to_lms[19]_i_2_n_0 ),
        .O(\TX_data_b_to_lms[19]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[19]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[51] ),
        .I2(TX_data_b[19]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[19]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_b_to_lms[1]_i_1 
       (.I0(TX_test_data_b[1]),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_b_to_lms[1]_i_2_n_0 ),
        .O(\TX_data_b_to_lms[1]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[1]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[33] ),
        .I2(TX_data_b[1]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[1]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[20]_i_1 
       (.I0(\TX_data_b_to_lms[20]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[20]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[20]),
        .O(\TX_data_b_to_lms[20]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[20]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[52] ),
        .I2(TX_data_b[20]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[20]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[21]_i_1 
       (.I0(\TX_data_b_to_lms[21]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[21]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[21]),
        .O(\TX_data_b_to_lms[21]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[21]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[53] ),
        .I2(TX_data_b[21]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[21]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[22]_i_1 
       (.I0(\TX_data_b_to_lms[22]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[22]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[22]),
        .O(\TX_data_b_to_lms[22]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[22]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[54] ),
        .I2(TX_data_b[22]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[22]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[23]_i_1 
       (.I0(\TX_data_b_to_lms[23]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[23]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[23]),
        .O(\TX_data_b_to_lms[23]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[23]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[55] ),
        .I2(TX_data_b[23]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[23]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[24]_i_1 
       (.I0(\TX_data_b_to_lms[24]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[24]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[24]),
        .O(\TX_data_b_to_lms[24]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[24]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[56] ),
        .I2(TX_data_b[24]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[24]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[25]_i_1 
       (.I0(\TX_data_b_to_lms[25]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[25]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[25]),
        .O(\TX_data_b_to_lms[25]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[25]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[57] ),
        .I2(TX_data_b[25]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[25]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[26]_i_1 
       (.I0(\TX_data_b_to_lms[26]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[26]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[26]),
        .O(\TX_data_b_to_lms[26]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[26]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[58] ),
        .I2(TX_data_b[26]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[26]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[27]_i_1 
       (.I0(\TX_data_b_to_lms[27]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[27]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[27]),
        .O(\TX_data_b_to_lms[27]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[27]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[59] ),
        .I2(TX_data_b[27]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[27]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[28]_i_1 
       (.I0(\TX_data_b_to_lms[28]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[28]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[28]),
        .O(\TX_data_b_to_lms[28]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[28]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[60] ),
        .I2(TX_data_b[28]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[28]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[29]_i_1 
       (.I0(\TX_data_b_to_lms[29]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[29]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[29]),
        .O(\TX_data_b_to_lms[29]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[29]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[61] ),
        .I2(TX_data_b[29]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[29]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_b_to_lms[2]_i_1 
       (.I0(TX_test_data_b[2]),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_b_to_lms[2]_i_2_n_0 ),
        .O(\TX_data_b_to_lms[2]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[2]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[34] ),
        .I2(TX_data_b[2]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[2]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[30]_i_1 
       (.I0(\TX_data_b_to_lms[30]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[30]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[30]),
        .O(\TX_data_b_to_lms[30]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[30]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[62] ),
        .I2(TX_data_b[30]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[30]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[31]_i_1 
       (.I0(\TX_data_b_to_lms[31]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[31]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[31]),
        .O(\TX_data_b_to_lms[31]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[31]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[63] ),
        .I2(TX_data_b[31]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[31]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFF08)) 
    \TX_data_b_to_lms[3]_i_1 
       (.I0(TX_test_data_b[3]),
        .I1(tx_test_mode_xfer_n_2),
        .I2(tx_test_mode_xfer_n_1),
        .I3(\TX_data_b_to_lms[3]_i_2_n_0 ),
        .O(\TX_data_b_to_lms[3]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[3]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[35] ),
        .I2(TX_data_b[3]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[3]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[4]_i_1 
       (.I0(\TX_data_b_to_lms[4]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[4]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[4]),
        .O(\TX_data_b_to_lms[4]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[4]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[36] ),
        .I2(TX_data_b[4]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[4]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[5]_i_1 
       (.I0(\TX_data_b_to_lms[5]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[5]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[5]),
        .O(\TX_data_b_to_lms[5]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[5]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[37] ),
        .I2(TX_data_b[5]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[5]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[6]_i_1 
       (.I0(\TX_data_b_to_lms[6]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[6]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[6]),
        .O(\TX_data_b_to_lms[6]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[6]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[38] ),
        .I2(TX_data_b[6]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[6]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[7]_i_1 
       (.I0(\TX_data_b_to_lms[7]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[7]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[7]),
        .O(\TX_data_b_to_lms[7]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[7]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[39] ),
        .I2(TX_data_b[7]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[7]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[8]_i_1 
       (.I0(\TX_data_b_to_lms[8]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[8]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[8]),
        .O(\TX_data_b_to_lms[8]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[8]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[40] ),
        .I2(TX_data_b[8]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[8]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAAAEA)) 
    \TX_data_b_to_lms[9]_i_1 
       (.I0(\TX_data_b_to_lms[9]_i_2_n_0 ),
        .I1(tx_test_mode_xfer_n_2),
        .I2(TX_test_data_b[9]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(RX_data_b_from_lms[9]),
        .O(\TX_data_b_to_lms[9]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00000066000000F0)) 
    \TX_data_b_to_lms[9]_i_2 
       (.I0(prbs_ctrl[13]),
        .I1(\u_tester/u_gen/out_reg_n_0_[41] ),
        .I2(TX_data_b[9]),
        .I3(tx_test_mode_xfer_n_1),
        .I4(tx_test_mode_xfer_n_2),
        .I5(PRBS_test_signal),
        .O(\TX_data_b_to_lms[9]_i_2_n_0 ));
  FDRE \TX_data_b_to_lms_reg[0] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[0]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[0]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[10] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[10]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[10]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[11] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[11]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[11]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[12] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[12]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[12]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[13] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[13]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[13]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[14] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[14]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[14]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[15] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[15]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[15]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[16] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[16]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[16]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[17] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[17]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[17]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[18] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[18]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[18]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[19] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[19]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[19]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[1] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[1]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[1]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[20] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[20]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[20]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[21] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[21]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[21]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[22] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[22]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[22]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[23] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[23]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[23]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[24] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[24]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[24]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[25] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[25]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[25]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[26] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[26]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[26]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[27] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[27]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[27]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[28] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[28]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[28]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[29] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[29]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[29]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[2] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[2]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[2]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[30] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[30]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[30]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[31] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[31]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[31]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[3] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[3]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[3]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[4] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[4]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[4]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[5] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[5]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[5]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[6] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[6]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[6]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[7] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[7]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[7]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[8] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[8]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[8]),
        .R(1'b0));
  FDRE \TX_data_b_to_lms_reg[9] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\TX_data_b_to_lms[9]_i_1_n_0 ),
        .Q(TX_data_b_to_lms[9]),
        .R(1'b0));
  VCC VCC
       (.P(\<const1> ));
  LUT1 #(
    .INIT(2'h1)) 
    \count[0]_i_1 
       (.I0(\u_tester/count_reg__0 [0]),
        .O(p_0_in__0[0]));
  (* SOFT_HLUTNM = "soft_lutpair43" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \count[1]_i_1 
       (.I0(\u_tester/count_reg__0 [0]),
        .I1(\u_tester/count_reg__0 [1]),
        .O(p_0_in__0[1]));
  (* SOFT_HLUTNM = "soft_lutpair43" *) 
  LUT3 #(
    .INIT(8'h78)) 
    \count[2]_i_1 
       (.I0(\u_tester/count_reg__0 [1]),
        .I1(\u_tester/count_reg__0 [0]),
        .I2(\u_tester/count_reg__0 [2]),
        .O(p_0_in__0[2]));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT4 #(
    .INIT(16'h7F80)) 
    \count[3]_i_1 
       (.I0(\u_tester/count_reg__0 [2]),
        .I1(\u_tester/count_reg__0 [0]),
        .I2(\u_tester/count_reg__0 [1]),
        .I3(\u_tester/count_reg__0 [3]),
        .O(p_0_in__0[3]));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT5 #(
    .INIT(32'h7FFF8000)) 
    \count[4]_i_1 
       (.I0(\u_tester/count_reg__0 [3]),
        .I1(\u_tester/count_reg__0 [1]),
        .I2(\u_tester/count_reg__0 [0]),
        .I3(\u_tester/count_reg__0 [2]),
        .I4(\u_tester/count_reg__0 [4]),
        .O(p_0_in__0[4]));
  LUT6 #(
    .INIT(64'h7FFFFFFF80000000)) 
    \count[5]_i_1 
       (.I0(\u_tester/count_reg__0 [4]),
        .I1(\u_tester/count_reg__0 [2]),
        .I2(\u_tester/count_reg__0 [0]),
        .I3(\u_tester/count_reg__0 [1]),
        .I4(\u_tester/count_reg__0 [3]),
        .I5(\u_tester/count_reg__0 [5]),
        .O(p_0_in__0[5]));
  (* SOFT_HLUTNM = "soft_lutpair56" *) 
  LUT2 #(
    .INIT(4'h9)) 
    \count[6]_i_1 
       (.I0(\count[6]_i_2_n_0 ),
        .I1(\u_tester/count_reg__0 [6]),
        .O(p_0_in__0[6]));
  LUT6 #(
    .INIT(64'h7FFFFFFFFFFFFFFF)) 
    \count[6]_i_2 
       (.I0(\u_tester/count_reg__0 [4]),
        .I1(\u_tester/count_reg__0 [2]),
        .I2(\u_tester/count_reg__0 [0]),
        .I3(\u_tester/count_reg__0 [1]),
        .I4(\u_tester/count_reg__0 [3]),
        .I5(\u_tester/count_reg__0 [5]),
        .O(\count[6]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h2)) 
    \counterFMax[7]_i_1 
       (.I0(\u_clocks/clk_hist [1]),
        .I1(\u_clocks/clk_hist [0]),
        .O(\counterFMax[7]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT1 #(
    .INIT(2'h1)) 
    \counterF[0]_i_1 
       (.I0(counterF[0]),
        .O(\counterF[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair38" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counterF[1]_i_1 
       (.I0(counterF[0]),
        .I1(counterF[1]),
        .O(\counterF[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair38" *) 
  LUT3 #(
    .INIT(8'h78)) 
    \counterF[2]_i_1 
       (.I0(counterF[0]),
        .I1(counterF[1]),
        .I2(counterF[2]),
        .O(\counterF[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT4 #(
    .INIT(16'h7F80)) 
    \counterF[3]_i_1 
       (.I0(counterF[1]),
        .I1(counterF[0]),
        .I2(counterF[2]),
        .I3(counterF[3]),
        .O(\counterF[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT5 #(
    .INIT(32'h7FFF8000)) 
    \counterF[4]_i_1 
       (.I0(counterF[2]),
        .I1(counterF[0]),
        .I2(counterF[1]),
        .I3(counterF[3]),
        .I4(counterF[4]),
        .O(\counterF[4]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h7FFFFFFF80000000)) 
    \counterF[5]_i_1 
       (.I0(counterF[3]),
        .I1(counterF[1]),
        .I2(counterF[0]),
        .I3(counterF[2]),
        .I4(counterF[4]),
        .I5(counterF[5]),
        .O(\counterF[5]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair37" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counterF[6]_i_1 
       (.I0(\counterF[7]_i_2_n_0 ),
        .I1(counterF[6]),
        .O(\counterF[6]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair37" *) 
  LUT3 #(
    .INIT(8'h78)) 
    \counterF[7]_i_1 
       (.I0(\counterF[7]_i_2_n_0 ),
        .I1(counterF[6]),
        .I2(counterF[7]),
        .O(\counterF[7]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h8000000000000000)) 
    \counterF[7]_i_2 
       (.I0(counterF[5]),
        .I1(counterF[3]),
        .I2(counterF[1]),
        .I3(counterF[0]),
        .I4(counterF[2]),
        .I5(counterF[4]),
        .O(\counterF[7]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h2)) 
    \counterRMax[7]_i_1 
       (.I0(\u_clocks/clk_hist [0]),
        .I1(\u_clocks/clk_hist [1]),
        .O(\counterRMax[7]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT1 #(
    .INIT(2'h1)) 
    \counterR[0]_i_1 
       (.I0(counterR[0]),
        .O(plusOp[0]));
  (* SOFT_HLUTNM = "soft_lutpair39" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counterR[1]_i_1 
       (.I0(counterR[0]),
        .I1(counterR[1]),
        .O(plusOp[1]));
  (* SOFT_HLUTNM = "soft_lutpair39" *) 
  LUT3 #(
    .INIT(8'h78)) 
    \counterR[2]_i_1 
       (.I0(counterR[0]),
        .I1(counterR[1]),
        .I2(counterR[2]),
        .O(plusOp[2]));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT4 #(
    .INIT(16'h7F80)) 
    \counterR[3]_i_1 
       (.I0(counterR[1]),
        .I1(counterR[0]),
        .I2(counterR[2]),
        .I3(counterR[3]),
        .O(plusOp[3]));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT5 #(
    .INIT(32'h7FFF8000)) 
    \counterR[4]_i_1 
       (.I0(counterR[2]),
        .I1(counterR[0]),
        .I2(counterR[1]),
        .I3(counterR[3]),
        .I4(counterR[4]),
        .O(plusOp[4]));
  LUT6 #(
    .INIT(64'h7FFFFFFF80000000)) 
    \counterR[5]_i_1 
       (.I0(counterR[3]),
        .I1(counterR[1]),
        .I2(counterR[0]),
        .I3(counterR[2]),
        .I4(counterR[4]),
        .I5(counterR[5]),
        .O(plusOp[5]));
  (* SOFT_HLUTNM = "soft_lutpair36" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counterR[6]_i_1 
       (.I0(\counterR[7]_i_2_n_0 ),
        .I1(counterR[6]),
        .O(plusOp[6]));
  (* SOFT_HLUTNM = "soft_lutpair36" *) 
  LUT3 #(
    .INIT(8'h78)) 
    \counterR[7]_i_1 
       (.I0(\counterR[7]_i_2_n_0 ),
        .I1(counterR[6]),
        .I2(counterR[7]),
        .O(plusOp[7]));
  LUT6 #(
    .INIT(64'h8000000000000000)) 
    \counterR[7]_i_2 
       (.I0(counterR[5]),
        .I1(counterR[3]),
        .I2(counterR[1]),
        .I3(counterR[0]),
        .I4(counterR[2]),
        .I5(counterR[4]),
        .O(\counterR[7]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \counter_src[0]_i_2 
       (.I0(\tx_cycle_counter/counter_src_reg [0]),
        .O(\counter_src[0]_i_2_n_0 ));
  CARRY4 \counter_src_reg[0]_i_1 
       (.CI(1'b0),
        .CO({\counter_src_reg[0]_i_1_n_0 ,\counter_src_reg[0]_i_1_n_1 ,\counter_src_reg[0]_i_1_n_2 ,\counter_src_reg[0]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b1}),
        .O({\counter_src_reg[0]_i_1_n_4 ,\counter_src_reg[0]_i_1_n_5 ,\counter_src_reg[0]_i_1_n_6 ,\counter_src_reg[0]_i_1_n_7 }),
        .S({\tx_cycle_counter/counter_src_reg [3:1],\counter_src[0]_i_2_n_0 }));
  CARRY4 \counter_src_reg[12]_i_1 
       (.CI(\counter_src_reg[8]_i_1_n_0 ),
        .CO({\counter_src_reg[12]_i_1_n_0 ,\counter_src_reg[12]_i_1_n_1 ,\counter_src_reg[12]_i_1_n_2 ,\counter_src_reg[12]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\counter_src_reg[12]_i_1_n_4 ,\counter_src_reg[12]_i_1_n_5 ,\counter_src_reg[12]_i_1_n_6 ,\counter_src_reg[12]_i_1_n_7 }),
        .S(\tx_cycle_counter/counter_src_reg [15:12]));
  CARRY4 \counter_src_reg[16]_i_1 
       (.CI(\counter_src_reg[12]_i_1_n_0 ),
        .CO({\counter_src_reg[16]_i_1_n_0 ,\counter_src_reg[16]_i_1_n_1 ,\counter_src_reg[16]_i_1_n_2 ,\counter_src_reg[16]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\counter_src_reg[16]_i_1_n_4 ,\counter_src_reg[16]_i_1_n_5 ,\counter_src_reg[16]_i_1_n_6 ,\counter_src_reg[16]_i_1_n_7 }),
        .S(\tx_cycle_counter/counter_src_reg [19:16]));
  CARRY4 \counter_src_reg[20]_i_1 
       (.CI(\counter_src_reg[16]_i_1_n_0 ),
        .CO({\counter_src_reg[20]_i_1_n_0 ,\counter_src_reg[20]_i_1_n_1 ,\counter_src_reg[20]_i_1_n_2 ,\counter_src_reg[20]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\counter_src_reg[20]_i_1_n_4 ,\counter_src_reg[20]_i_1_n_5 ,\counter_src_reg[20]_i_1_n_6 ,\counter_src_reg[20]_i_1_n_7 }),
        .S(\tx_cycle_counter/counter_src_reg [23:20]));
  CARRY4 \counter_src_reg[24]_i_1 
       (.CI(\counter_src_reg[20]_i_1_n_0 ),
        .CO({\counter_src_reg[24]_i_1_n_0 ,\counter_src_reg[24]_i_1_n_1 ,\counter_src_reg[24]_i_1_n_2 ,\counter_src_reg[24]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\counter_src_reg[24]_i_1_n_4 ,\counter_src_reg[24]_i_1_n_5 ,\counter_src_reg[24]_i_1_n_6 ,\counter_src_reg[24]_i_1_n_7 }),
        .S(\tx_cycle_counter/counter_src_reg [27:24]));
  CARRY4 \counter_src_reg[28]_i_1 
       (.CI(\counter_src_reg[24]_i_1_n_0 ),
        .CO({\NLW_counter_src_reg[28]_i_1_CO_UNCONNECTED [3],\counter_src_reg[28]_i_1_n_1 ,\counter_src_reg[28]_i_1_n_2 ,\counter_src_reg[28]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\counter_src_reg[28]_i_1_n_4 ,\counter_src_reg[28]_i_1_n_5 ,\counter_src_reg[28]_i_1_n_6 ,\counter_src_reg[28]_i_1_n_7 }),
        .S(\tx_cycle_counter/counter_src_reg [31:28]));
  CARRY4 \counter_src_reg[4]_i_1 
       (.CI(\counter_src_reg[0]_i_1_n_0 ),
        .CO({\counter_src_reg[4]_i_1_n_0 ,\counter_src_reg[4]_i_1_n_1 ,\counter_src_reg[4]_i_1_n_2 ,\counter_src_reg[4]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\counter_src_reg[4]_i_1_n_4 ,\counter_src_reg[4]_i_1_n_5 ,\counter_src_reg[4]_i_1_n_6 ,\counter_src_reg[4]_i_1_n_7 }),
        .S(\tx_cycle_counter/counter_src_reg [7:4]));
  CARRY4 \counter_src_reg[8]_i_1 
       (.CI(\counter_src_reg[4]_i_1_n_0 ),
        .CO({\counter_src_reg[8]_i_1_n_0 ,\counter_src_reg[8]_i_1_n_1 ,\counter_src_reg[8]_i_1_n_2 ,\counter_src_reg[8]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\counter_src_reg[8]_i_1_n_4 ,\counter_src_reg[8]_i_1_n_5 ,\counter_src_reg[8]_i_1_n_6 ,\counter_src_reg[8]_i_1_n_7 }),
        .S(\tx_cycle_counter/counter_src_reg [11:8]));
  LUT1 #(
    .INIT(2'h1)) 
    data_stb_i_i_1
       (.I0(\trxiq_tx/data_stb_i_reg_n_0 ),
        .O(data_stb_i_i_1_n_0));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[10]_i_1 
       (.I0(prbs_e[10]),
        .I1(RX_data_a[10]),
        .I2(\u_tester/chk_out [10]),
        .O(\err[10]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[11]_i_1 
       (.I0(prbs_e[11]),
        .I1(RX_data_a[11]),
        .I2(\u_tester/chk_out [11]),
        .O(\err[11]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[12]_i_1 
       (.I0(prbs_e[12]),
        .I1(RX_data_a[12]),
        .I2(\u_tester/chk_out [12]),
        .O(\err[12]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[13]_i_1 
       (.I0(prbs_e[13]),
        .I1(RX_data_a[13]),
        .I2(\u_tester/chk_out [13]),
        .O(\err[13]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[14]_i_1 
       (.I0(prbs_e[14]),
        .I1(RX_data_a[14]),
        .I2(\u_tester/chk_out [14]),
        .O(\err[14]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hF6)) 
    \err[15]_i_1 
       (.I0(RX_data_a[15]),
        .I1(\u_tester/chk_out [15]),
        .I2(prbs_e[15]),
        .O(\err[15]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hF6)) 
    \err[20]_i_1 
       (.I0(RX_data_a[20]),
        .I1(\u_tester/chk_out [20]),
        .I2(prbs_e[20]),
        .O(\err[20]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[21]_i_1 
       (.I0(prbs_e[21]),
        .I1(RX_data_a[21]),
        .I2(\u_tester/chk_out [21]),
        .O(\err[21]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[22]_i_1 
       (.I0(prbs_e[22]),
        .I1(RX_data_a[22]),
        .I2(\u_tester/chk_out [22]),
        .O(\err[22]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[23]_i_1 
       (.I0(prbs_e[23]),
        .I1(RX_data_a[23]),
        .I2(\u_tester/chk_out [23]),
        .O(\err[23]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[24]_i_1 
       (.I0(prbs_e[24]),
        .I1(RX_data_a[24]),
        .I2(\u_tester/chk_out [24]),
        .O(\err[24]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[25]_i_1 
       (.I0(prbs_e[25]),
        .I1(RX_data_a[25]),
        .I2(\u_tester/chk_out [25]),
        .O(\err[25]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[26]_i_1 
       (.I0(prbs_e[26]),
        .I1(RX_data_a[26]),
        .I2(\u_tester/chk_out [26]),
        .O(\err[26]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[27]_i_1 
       (.I0(prbs_e[27]),
        .I1(RX_data_a[27]),
        .I2(\u_tester/chk_out [27]),
        .O(\err[27]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[28]_i_1 
       (.I0(prbs_e[28]),
        .I1(RX_data_a[28]),
        .I2(\u_tester/chk_out [28]),
        .O(\err[28]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[29]_i_1 
       (.I0(prbs_e[29]),
        .I1(RX_data_a[29]),
        .I2(\u_tester/chk_out [29]),
        .O(\err[29]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hF6)) 
    \err[30]_i_1 
       (.I0(RX_data_a[30]),
        .I1(\u_tester/chk_out [30]),
        .I2(prbs_e[30]),
        .O(\err[30]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[31]_i_1 
       (.I0(prbs_e[31]),
        .I1(RX_data_a[31]),
        .I2(\u_tester/chk_out [31]),
        .O(\err[31]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[36]_i_1 
       (.I0(prbs_e[36]),
        .I1(RX_data_b[4]),
        .I2(\u_tester/chk_out [36]),
        .O(\err[36]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[37]_i_1 
       (.I0(prbs_e[37]),
        .I1(RX_data_b[5]),
        .I2(\u_tester/chk_out [37]),
        .O(\err[37]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[38]_i_1 
       (.I0(prbs_e[38]),
        .I1(RX_data_b[6]),
        .I2(\u_tester/chk_out [38]),
        .O(\err[38]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[39]_i_1 
       (.I0(prbs_e[39]),
        .I1(RX_data_b[7]),
        .I2(\u_tester/chk_out [39]),
        .O(\err[39]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[40]_i_1 
       (.I0(prbs_e[40]),
        .I1(RX_data_b[8]),
        .I2(\u_tester/chk_out [40]),
        .O(\err[40]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[41]_i_1 
       (.I0(prbs_e[41]),
        .I1(RX_data_b[9]),
        .I2(\u_tester/chk_out [41]),
        .O(\err[41]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[42]_i_1 
       (.I0(prbs_e[42]),
        .I1(RX_data_b[10]),
        .I2(\u_tester/chk_out [42]),
        .O(\err[42]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[43]_i_1 
       (.I0(prbs_e[43]),
        .I1(RX_data_b[11]),
        .I2(\u_tester/chk_out [43]),
        .O(\err[43]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[44]_i_1 
       (.I0(prbs_e[44]),
        .I1(RX_data_b[12]),
        .I2(\u_tester/chk_out [44]),
        .O(\err[44]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[45]_i_1 
       (.I0(prbs_e[45]),
        .I1(RX_data_b[13]),
        .I2(\u_tester/chk_out [45]),
        .O(\err[45]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[46]_i_1 
       (.I0(prbs_e[46]),
        .I1(RX_data_b[14]),
        .I2(\u_tester/chk_out [46]),
        .O(\err[46]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[47]_i_1 
       (.I0(prbs_e[47]),
        .I1(RX_data_b[15]),
        .I2(\u_tester/chk_out [47]),
        .O(\err[47]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hF6)) 
    \err[4]_i_1 
       (.I0(RX_data_a[4]),
        .I1(\u_tester/chk_out [4]),
        .I2(prbs_e[4]),
        .O(\err[4]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hF6)) 
    \err[52]_i_1 
       (.I0(RX_data_b[20]),
        .I1(\u_tester/chk_out [52]),
        .I2(prbs_e[52]),
        .O(\err[52]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[53]_i_1 
       (.I0(prbs_e[53]),
        .I1(RX_data_b[21]),
        .I2(\u_tester/chk_out [53]),
        .O(\err[53]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[54]_i_1 
       (.I0(prbs_e[54]),
        .I1(RX_data_b[22]),
        .I2(\u_tester/chk_out [54]),
        .O(\err[54]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[55]_i_1 
       (.I0(prbs_e[55]),
        .I1(RX_data_b[23]),
        .I2(\u_tester/chk_out [55]),
        .O(\err[55]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[56]_i_1 
       (.I0(prbs_e[56]),
        .I1(RX_data_b[24]),
        .I2(\u_tester/chk_out [56]),
        .O(\err[56]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[57]_i_1 
       (.I0(prbs_e[57]),
        .I1(RX_data_b[25]),
        .I2(\u_tester/chk_out [57]),
        .O(\err[57]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[58]_i_1 
       (.I0(prbs_e[58]),
        .I1(RX_data_b[26]),
        .I2(\u_tester/chk_out [58]),
        .O(\err[58]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[59]_i_1 
       (.I0(prbs_e[59]),
        .I1(RX_data_b[27]),
        .I2(\u_tester/chk_out [59]),
        .O(\err[59]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[5]_i_1 
       (.I0(prbs_e[5]),
        .I1(RX_data_a[5]),
        .I2(\u_tester/chk_out [5]),
        .O(\err[5]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[60]_i_1 
       (.I0(prbs_e[60]),
        .I1(RX_data_b[28]),
        .I2(\u_tester/chk_out [60]),
        .O(\err[60]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[61]_i_1 
       (.I0(prbs_e[61]),
        .I1(RX_data_b[29]),
        .I2(\u_tester/chk_out [61]),
        .O(\err[61]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[62]_i_1 
       (.I0(prbs_e[62]),
        .I1(RX_data_b[30]),
        .I2(\u_tester/chk_out [62]),
        .O(\err[62]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'hFFF1)) 
    \err[63]_i_1 
       (.I0(\u_tester/st_cur [0]),
        .I1(\u_tester/st_cur [1]),
        .I2(DATA_rst),
        .I3(prbs_ctrl[14]),
        .O(\err[63]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hF6)) 
    \err[63]_i_2 
       (.I0(\u_tester/chk_out [63]),
        .I1(RX_data_b[31]),
        .I2(prbs_e[63]),
        .O(\err[63]_i_2_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[6]_i_1 
       (.I0(prbs_e[6]),
        .I1(RX_data_a[6]),
        .I2(\u_tester/chk_out [6]),
        .O(\err[6]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[7]_i_1 
       (.I0(prbs_e[7]),
        .I1(RX_data_a[7]),
        .I2(\u_tester/chk_out [7]),
        .O(\err[7]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[8]_i_1 
       (.I0(prbs_e[8]),
        .I1(RX_data_a[8]),
        .I2(\u_tester/chk_out [8]),
        .O(\err[8]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBE)) 
    \err[9]_i_1 
       (.I0(prbs_e[9]),
        .I1(RX_data_a[9]),
        .I2(\u_tester/chk_out [9]),
        .O(\err[9]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \mask_count[63]_i_1 
       (.I0(DATA_rst),
        .I1(prbs_ctrl[15]),
        .O(\u_tester/mask_count ));
  LUT4 #(
    .INIT(16'h9669)) 
    \out[0]_i_1 
       (.I0(\u_tester/u_gen/out_reg_n_0_[60] ),
        .I1(\u_tester/u_gen/out_reg_n_0_[59] ),
        .I2(\u_tester/u_gen/out_reg_n_0_[63] ),
        .I3(\u_tester/u_gen/out_reg_n_0_[62] ),
        .O(p_0_out));
  LUT3 #(
    .INIT(8'hEF)) 
    \out[63]_i_1 
       (.I0(prbs_ctrl[15]),
        .I1(DATA_rst),
        .I2(prbs_ctrl[12]),
        .O(\out[63]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \out[63]_i_1__0 
       (.I0(\u_tester/st_cur [0]),
        .I1(\u_tester/st_cur [1]),
        .O(\u_tester/chk_en ));
  LUT4 #(
    .INIT(16'h9669)) 
    \out_reg[2]_srl3___u_tester_u_chk_out_reg_r_1_i_1 
       (.I0(\u_tester/chk_out [60]),
        .I1(\u_tester/chk_out [59]),
        .I2(\u_tester/chk_out [63]),
        .I3(\u_tester/chk_out [62]),
        .O(\out_reg[2]_srl3___u_tester_u_chk_out_reg_r_1_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry__0_i_1
       (.I0(\u_tester/chk_out [26]),
        .I1(RX_data_a[26]),
        .I2(\u_tester/chk_out [25]),
        .I3(RX_data_a[25]),
        .I4(RX_data_a[24]),
        .I5(\u_tester/chk_out [24]),
        .O(rx_err0_carry__0_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry__0_i_2
       (.I0(\u_tester/chk_out [23]),
        .I1(RX_data_a[23]),
        .I2(\u_tester/chk_out [22]),
        .I3(RX_data_a[22]),
        .I4(RX_data_a[21]),
        .I5(\u_tester/chk_out [21]),
        .O(rx_err0_carry__0_i_2_n_0));
  LUT2 #(
    .INIT(4'h9)) 
    rx_err0_carry__0_i_3
       (.I0(RX_data_a[20]),
        .I1(\u_tester/chk_out [20]),
        .O(rx_err0_carry__0_i_3_n_0));
  LUT2 #(
    .INIT(4'h9)) 
    rx_err0_carry__0_i_4
       (.I0(RX_data_a[15]),
        .I1(\u_tester/chk_out [15]),
        .O(rx_err0_carry__0_i_4_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry__1_i_1
       (.I0(\u_tester/chk_out [41]),
        .I1(RX_data_b[9]),
        .I2(\u_tester/chk_out [40]),
        .I3(RX_data_b[8]),
        .I4(RX_data_b[7]),
        .I5(\u_tester/chk_out [39]),
        .O(rx_err0_carry__1_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry__1_i_2
       (.I0(\u_tester/chk_out [38]),
        .I1(RX_data_b[6]),
        .I2(\u_tester/chk_out [37]),
        .I3(RX_data_b[5]),
        .I4(RX_data_b[4]),
        .I5(\u_tester/chk_out [36]),
        .O(rx_err0_carry__1_i_2_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    rx_err0_carry__1_i_3
       (.I0(\u_tester/chk_out [31]),
        .I1(RX_data_a[31]),
        .I2(\u_tester/chk_out [30]),
        .I3(RX_data_a[30]),
        .O(rx_err0_carry__1_i_3_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry__1_i_4
       (.I0(\u_tester/chk_out [29]),
        .I1(RX_data_a[29]),
        .I2(\u_tester/chk_out [28]),
        .I3(RX_data_a[28]),
        .I4(RX_data_a[27]),
        .I5(\u_tester/chk_out [27]),
        .O(rx_err0_carry__1_i_4_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry__2_i_1
       (.I0(\u_tester/chk_out [56]),
        .I1(RX_data_b[24]),
        .I2(\u_tester/chk_out [55]),
        .I3(RX_data_b[23]),
        .I4(RX_data_b[22]),
        .I5(\u_tester/chk_out [54]),
        .O(rx_err0_carry__2_i_1_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    rx_err0_carry__2_i_2
       (.I0(\u_tester/chk_out [53]),
        .I1(RX_data_b[21]),
        .I2(\u_tester/chk_out [52]),
        .I3(RX_data_b[20]),
        .O(rx_err0_carry__2_i_2_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry__2_i_3
       (.I0(\u_tester/chk_out [47]),
        .I1(RX_data_b[15]),
        .I2(\u_tester/chk_out [46]),
        .I3(RX_data_b[14]),
        .I4(RX_data_b[13]),
        .I5(\u_tester/chk_out [45]),
        .O(rx_err0_carry__2_i_3_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry__2_i_4
       (.I0(\u_tester/chk_out [44]),
        .I1(RX_data_b[12]),
        .I2(\u_tester/chk_out [43]),
        .I3(RX_data_b[11]),
        .I4(RX_data_b[10]),
        .I5(\u_tester/chk_out [42]),
        .O(rx_err0_carry__2_i_4_n_0));
  LUT2 #(
    .INIT(4'h9)) 
    rx_err0_carry__3_i_1
       (.I0(\u_tester/chk_out [63]),
        .I1(RX_data_b[31]),
        .O(rx_err0_carry__3_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry__3_i_2
       (.I0(\u_tester/chk_out [62]),
        .I1(RX_data_b[30]),
        .I2(\u_tester/chk_out [61]),
        .I3(RX_data_b[29]),
        .I4(RX_data_b[28]),
        .I5(\u_tester/chk_out [60]),
        .O(rx_err0_carry__3_i_2_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry__3_i_3
       (.I0(\u_tester/chk_out [59]),
        .I1(RX_data_b[27]),
        .I2(\u_tester/chk_out [58]),
        .I3(RX_data_b[26]),
        .I4(RX_data_b[25]),
        .I5(\u_tester/chk_out [57]),
        .O(rx_err0_carry__3_i_3_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry_i_1
       (.I0(\u_tester/chk_out [14]),
        .I1(RX_data_a[14]),
        .I2(\u_tester/chk_out [13]),
        .I3(RX_data_a[13]),
        .I4(RX_data_a[12]),
        .I5(\u_tester/chk_out [12]),
        .O(rx_err0_carry_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry_i_2
       (.I0(\u_tester/chk_out [11]),
        .I1(RX_data_a[11]),
        .I2(\u_tester/chk_out [10]),
        .I3(RX_data_a[10]),
        .I4(RX_data_a[9]),
        .I5(\u_tester/chk_out [9]),
        .O(rx_err0_carry_i_2_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    rx_err0_carry_i_3
       (.I0(\u_tester/chk_out [8]),
        .I1(RX_data_a[8]),
        .I2(\u_tester/chk_out [7]),
        .I3(RX_data_a[7]),
        .I4(RX_data_a[6]),
        .I5(\u_tester/chk_out [6]),
        .O(rx_err0_carry_i_3_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    rx_err0_carry_i_4
       (.I0(\u_tester/chk_out [5]),
        .I1(RX_data_a[5]),
        .I2(\u_tester/chk_out [4]),
        .I3(RX_data_a[4]),
        .O(rx_err0_carry_i_4_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    \rx_err_cnt[0]_i_1 
       (.I0(prbs_stat[0]),
        .O(p_0_in__1[0]));
  (* SOFT_HLUTNM = "soft_lutpair46" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \rx_err_cnt[1]_i_1 
       (.I0(prbs_stat[0]),
        .I1(prbs_stat[1]),
        .O(p_0_in__1[1]));
  (* SOFT_HLUTNM = "soft_lutpair46" *) 
  LUT3 #(
    .INIT(8'h78)) 
    \rx_err_cnt[2]_i_1 
       (.I0(prbs_stat[1]),
        .I1(prbs_stat[0]),
        .I2(prbs_stat[2]),
        .O(p_0_in__1[2]));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT4 #(
    .INIT(16'h7F80)) 
    \rx_err_cnt[3]_i_1 
       (.I0(prbs_stat[2]),
        .I1(prbs_stat[0]),
        .I2(prbs_stat[1]),
        .I3(prbs_stat[3]),
        .O(p_0_in__1[3]));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT5 #(
    .INIT(32'h7FFF8000)) 
    \rx_err_cnt[4]_i_1 
       (.I0(prbs_stat[3]),
        .I1(prbs_stat[1]),
        .I2(prbs_stat[0]),
        .I3(prbs_stat[2]),
        .I4(prbs_stat[4]),
        .O(p_0_in__1[4]));
  LUT6 #(
    .INIT(64'h7FFFFFFF80000000)) 
    \rx_err_cnt[5]_i_1 
       (.I0(prbs_stat[4]),
        .I1(prbs_stat[2]),
        .I2(prbs_stat[0]),
        .I3(prbs_stat[1]),
        .I4(prbs_stat[3]),
        .I5(prbs_stat[5]),
        .O(p_0_in__1[5]));
  (* SOFT_HLUTNM = "soft_lutpair47" *) 
  LUT2 #(
    .INIT(4'h9)) 
    \rx_err_cnt[6]_i_1 
       (.I0(\rx_err_cnt[7]_i_3_n_0 ),
        .I1(prbs_stat[6]),
        .O(p_0_in__1[6]));
  LUT6 #(
    .INIT(64'hA8A800A8A8A8A8A8)) 
    \rx_err_cnt[7]_i_1 
       (.I0(prbs_stat[21]),
        .I1(\u_tester/st_cur [0]),
        .I2(\u_tester/st_cur [1]),
        .I3(prbs_stat[7]),
        .I4(\rx_err_cnt[7]_i_3_n_0 ),
        .I5(prbs_stat[6]),
        .O(\u_tester/rx_err_cnt0 ));
  (* SOFT_HLUTNM = "soft_lutpair47" *) 
  LUT3 #(
    .INIT(8'hD2)) 
    \rx_err_cnt[7]_i_2 
       (.I0(prbs_stat[6]),
        .I1(\rx_err_cnt[7]_i_3_n_0 ),
        .I2(prbs_stat[7]),
        .O(p_0_in__1[7]));
  LUT6 #(
    .INIT(64'h7FFFFFFFFFFFFFFF)) 
    \rx_err_cnt[7]_i_3 
       (.I0(prbs_stat[4]),
        .I1(prbs_stat[2]),
        .I2(prbs_stat[0]),
        .I3(prbs_stat[1]),
        .I4(prbs_stat[3]),
        .I5(prbs_stat[5]),
        .O(\rx_err_cnt[7]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h000E000E000E0000)) 
    rx_err_s_i_1
       (.I0(\u_tester/st_cur [0]),
        .I1(\u_tester/st_cur [1]),
        .I2(DATA_rst),
        .I3(prbs_ctrl[14]),
        .I4(prbs_stat[21]),
        .I5(\u_tester/rx_err0 ),
        .O(rx_err_s_i_1_n_0));
  (* DEF_VAL = "1'b1" *) 
  (* DEST_SYNC_FF = "4" *) 
  (* INIT = "1" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* VERSION = "0" *) 
  (* XPM_CDC = "SYNC_RST" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_sync_rst rx_rst_gen
       (.dest_clk(DATA_clk),
        .dest_rst(DATA_rst),
        .src_rst(EXT_rst));
  (* DEST_SYNC_FF = "4" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_INPUT_REG = "1" *) 
  (* VERSION = "0" *) 
  (* WIDTH = "64" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_array_single__parameterized1 rx_test_data_xfer
       (.dest_clk(SYS_clk),
        .dest_out({SYS_test_data_b_rx,SYS_test_data_a_rx}),
        .src_clk(DATA_clk),
        .src_in({RX_data_b,RX_data_a}));
  (* DEST_SYNC_FF = "4" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_INPUT_REG = "1" *) 
  (* VERSION = "0" *) 
  (* WIDTH = "2" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_array_single rx_test_mode_xfer
       (.dest_clk(DATA_clk),
        .dest_out({rx_test_mode_xfer_n_0,rx_test_mode_xfer_n_1}),
        .src_clk(SYS_clk),
        .src_in(SYS_test_mode[1:0]));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT3 #(
    .INIT(8'hB4)) 
    samp_clk_1x_i_1
       (.I0(\u_clocks/clk_hist [1]),
        .I1(\u_clocks/clk_hist [0]),
        .I2(\u_clocks/samp_clk_1x ),
        .O(samp_clk_1x_i_1_n_0));
  LUT6 #(
    .INIT(64'h00000000EEEAEAEE)) 
    samp_clk_2x_i_1
       (.I0(\u_clocks/samp_clk_2x ),
        .I1(samp_clk_2x_i_2_n_0),
        .I2(\counterRMax[7]_i_1_n_0 ),
        .I3(\counterR[7]_i_2_n_0 ),
        .I4(counterR[6]),
        .I5(samp_clk_2x_i_3_n_0),
        .O(samp_clk_2x_i_1_n_0));
  LUT6 #(
    .INIT(64'h5535AA3AAA3A5535)) 
    samp_clk_2x_i_10
       (.I0(counterRMax[3]),
        .I1(counterR[3]),
        .I2(\u_clocks/clk_hist [0]),
        .I3(\u_clocks/clk_hist [1]),
        .I4(counterR[0]),
        .I5(counterR[1]),
        .O(samp_clk_2x_i_10_n_0));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT4 #(
    .INIT(16'hFB08)) 
    samp_clk_2x_i_11
       (.I0(counterR[7]),
        .I1(\u_clocks/clk_hist [0]),
        .I2(\u_clocks/clk_hist [1]),
        .I3(counterRMax[7]),
        .O(p_0_in));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT4 #(
    .INIT(16'hFB08)) 
    samp_clk_2x_i_12
       (.I0(counterF[7]),
        .I1(\u_clocks/clk_hist [1]),
        .I2(\u_clocks/clk_hist [0]),
        .I3(counterFMax[7]),
        .O(p_0_in0_in));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT5 #(
    .INIT(32'h5535AA3A)) 
    samp_clk_2x_i_13
       (.I0(counterFMax[2]),
        .I1(counterF[2]),
        .I2(\u_clocks/clk_hist [1]),
        .I3(\u_clocks/clk_hist [0]),
        .I4(counterF[0]),
        .O(samp_clk_2x_i_13_n_0));
  LUT6 #(
    .INIT(64'h5535AA3AAA3A5535)) 
    samp_clk_2x_i_14
       (.I0(counterFMax[3]),
        .I1(counterF[3]),
        .I2(\u_clocks/clk_hist [1]),
        .I3(\u_clocks/clk_hist [0]),
        .I4(counterF[0]),
        .I5(counterF[1]),
        .O(samp_clk_2x_i_14_n_0));
  LUT6 #(
    .INIT(64'h0088808000880808)) 
    samp_clk_2x_i_2
       (.I0(samp_clk_2x_i_4_n_0),
        .I1(samp_clk_2x_i_5_n_0),
        .I2(counterRMax[5]),
        .I3(counterR[5]),
        .I4(\counterRMax[7]_i_1_n_0 ),
        .I5(plusOp[3]),
        .O(samp_clk_2x_i_2_n_0));
  LUT6 #(
    .INIT(64'h8080800080008080)) 
    samp_clk_2x_i_3
       (.I0(samp_clk_2x_i_6_n_0),
        .I1(samp_clk_2x_i_7_n_0),
        .I2(samp_clk_2x_i_8_n_0),
        .I3(\counterFMax[7]_i_1_n_0 ),
        .I4(\counterF[7]_i_2_n_0 ),
        .I5(counterF[6]),
        .O(samp_clk_2x_i_3_n_0));
  LUT6 #(
    .INIT(64'h0A880A2200000000)) 
    samp_clk_2x_i_4
       (.I0(samp_clk_2x_i_9_n_0),
        .I1(counterRMax[4]),
        .I2(counterR[4]),
        .I3(\counterRMax[7]_i_1_n_0 ),
        .I4(plusOp[2]),
        .I5(samp_clk_2x_i_10_n_0),
        .O(samp_clk_2x_i_4_n_0));
  LUT6 #(
    .INIT(64'h0000A500333300A5)) 
    samp_clk_2x_i_5
       (.I0(plusOp[4]),
        .I1(counterR[6]),
        .I2(counterRMax[6]),
        .I3(plusOp[5]),
        .I4(\counterRMax[7]_i_1_n_0 ),
        .I5(p_0_in),
        .O(samp_clk_2x_i_5_n_0));
  LUT5 #(
    .INIT(32'hAA3A5535)) 
    samp_clk_2x_i_6
       (.I0(counterFMax[5]),
        .I1(counterF[5]),
        .I2(\u_clocks/clk_hist [1]),
        .I3(\u_clocks/clk_hist [0]),
        .I4(\counterF[3]_i_1_n_0 ),
        .O(samp_clk_2x_i_6_n_0));
  LUT6 #(
    .INIT(64'h0000A500333300A5)) 
    samp_clk_2x_i_7
       (.I0(\counterF[4]_i_1_n_0 ),
        .I1(counterF[6]),
        .I2(counterFMax[6]),
        .I3(\counterF[5]_i_1_n_0 ),
        .I4(\counterFMax[7]_i_1_n_0 ),
        .I5(p_0_in0_in),
        .O(samp_clk_2x_i_7_n_0));
  LUT6 #(
    .INIT(64'h0A880A2200000000)) 
    samp_clk_2x_i_8
       (.I0(samp_clk_2x_i_13_n_0),
        .I1(counterFMax[4]),
        .I2(counterF[4]),
        .I3(\counterFMax[7]_i_1_n_0 ),
        .I4(\counterF[2]_i_1_n_0 ),
        .I5(samp_clk_2x_i_14_n_0),
        .O(samp_clk_2x_i_8_n_0));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT5 #(
    .INIT(32'h5535AA3A)) 
    samp_clk_2x_i_9
       (.I0(counterRMax[2]),
        .I1(counterR[2]),
        .I2(\u_clocks/clk_hist [0]),
        .I3(\u_clocks/clk_hist [1]),
        .I4(counterR[0]),
        .O(samp_clk_2x_i_9_n_0));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT3 #(
    .INIT(8'hB2)) 
    samp_clk_fb_i_1
       (.I0(\u_clocks/samp_clk_fb_reg_n_0 ),
        .I1(\u_clocks/clk_hist [1]),
        .I2(\u_clocks/clk_hist [0]),
        .O(samp_clk_fb_i_1_n_0));
  LUT5 #(
    .INIT(32'h0000EA2A)) 
    sel_mmcm_i_1
       (.I0(sel_mmcm),
        .I1(SYS_config_data[29]),
        .I2(SYS_config_write),
        .I3(SYS_config_data[1]),
        .I4(SYS_rst),
        .O(sel_mmcm_i_1_n_0));
  FDRE sel_mmcm_reg
       (.C(SYS_clk),
        .CE(1'b1),
        .D(sel_mmcm_i_1_n_0),
        .Q(sel_mmcm),
        .R(1'b0));
  (* SOFT_HLUTNM = "soft_lutpair58" *) 
  LUT1 #(
    .INIT(2'h1)) 
    \slip_cnt[0]_i_1 
       (.I0(prbs_stat[16]),
        .O(p_0_in__2[0]));
  (* SOFT_HLUTNM = "soft_lutpair58" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \slip_cnt[1]_i_1 
       (.I0(prbs_stat[16]),
        .I1(prbs_stat[17]),
        .O(p_0_in__2[1]));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT3 #(
    .INIT(8'h78)) 
    \slip_cnt[2]_i_1 
       (.I0(prbs_stat[17]),
        .I1(prbs_stat[16]),
        .I2(prbs_stat[18]),
        .O(p_0_in__2[2]));
  LUT2 #(
    .INIT(4'hB)) 
    \slip_cnt[3]_i_1 
       (.I0(DATA_rst),
        .I1(prbs_ctrl[12]),
        .O(\slip_cnt[3]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h1)) 
    \slip_cnt[3]_i_2 
       (.I0(\u_tester/st_cur [1]),
        .I1(\u_tester/st_cur [0]),
        .O(\u_tester/rx_slip ));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT4 #(
    .INIT(16'h7F80)) 
    \slip_cnt[3]_i_3 
       (.I0(prbs_stat[18]),
        .I1(prbs_stat[16]),
        .I2(prbs_stat[17]),
        .I3(prbs_stat[19]),
        .O(p_0_in__2[3]));
  (* SOFT_HLUTNM = "soft_lutpair41" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \st_cur[0]_i_1 
       (.I0(\st_cur[1]_i_2_n_0 ),
        .I1(\st_cur[0]_i_2_n_0 ),
        .O(\st_cur[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h3030333320302030)) 
    \st_cur[0]_i_2 
       (.I0(\st_cur[1]_i_10_n_0 ),
        .I1(\out[63]_i_1_n_0 ),
        .I2(\u_tester/st_cur [0]),
        .I3(prbs_stat[21]),
        .I4(\u_tester/rx_err0 ),
        .I5(\u_tester/st_cur [1]),
        .O(\st_cur[0]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair41" *) 
  LUT3 #(
    .INIT(8'h0D)) 
    \st_cur[1]_i_1 
       (.I0(\st_cur[1]_i_2_n_0 ),
        .I1(\u_tester/st_nxt ),
        .I2(\out[63]_i_1_n_0 ),
        .O(\st_cur[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair56" *) 
  LUT2 #(
    .INIT(4'hB)) 
    \st_cur[1]_i_10 
       (.I0(\count[6]_i_2_n_0 ),
        .I1(\u_tester/count_reg__0 [6]),
        .O(\st_cur[1]_i_10_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \st_cur[1]_i_11 
       (.I0(\u_tester/mask_count_reg_n_0_[53] ),
        .I1(\u_tester/mask_count_reg_n_0_[52] ),
        .I2(\u_tester/mask_count_reg_n_0_[55] ),
        .I3(\u_tester/mask_count_reg_n_0_[54] ),
        .O(\st_cur[1]_i_11_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \st_cur[1]_i_12 
       (.I0(\u_tester/mask_count_reg_n_0_[61] ),
        .I1(\u_tester/mask_count_reg_n_0_[60] ),
        .I2(\u_tester/mask_count_reg_n_0_[63] ),
        .I3(\u_tester/mask_count_reg_n_0_[62] ),
        .O(\st_cur[1]_i_12_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \st_cur[1]_i_13 
       (.I0(\u_tester/mask_count_reg_n_0_[29] ),
        .I1(\u_tester/mask_count_reg_n_0_[28] ),
        .I2(\u_tester/mask_count_reg_n_0_[31] ),
        .I3(\u_tester/mask_count_reg_n_0_[30] ),
        .O(\st_cur[1]_i_13_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \st_cur[1]_i_14 
       (.I0(\u_tester/mask_count_reg_n_0_[41] ),
        .I1(\u_tester/mask_count_reg_n_0_[40] ),
        .I2(\u_tester/mask_count_reg_n_0_[43] ),
        .I3(\u_tester/mask_count_reg_n_0_[42] ),
        .O(\st_cur[1]_i_14_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \st_cur[1]_i_15 
       (.I0(\u_tester/mask_count_reg_n_0_[21] ),
        .I1(\u_tester/mask_count_reg_n_0_[20] ),
        .I2(\u_tester/mask_count_reg_n_0_[23] ),
        .I3(\u_tester/mask_count_reg_n_0_[22] ),
        .O(\st_cur[1]_i_15_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \st_cur[1]_i_16 
       (.I0(\u_tester/mask_count_reg_n_0_[9] ),
        .I1(\u_tester/mask_count_reg_n_0_[8] ),
        .I2(\u_tester/mask_count_reg_n_0_[11] ),
        .I3(\u_tester/mask_count_reg_n_0_[10] ),
        .O(\st_cur[1]_i_16_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFE)) 
    \st_cur[1]_i_2 
       (.I0(\st_cur[1]_i_4_n_0 ),
        .I1(\st_cur[1]_i_5_n_0 ),
        .I2(\st_cur[1]_i_6_n_0 ),
        .I3(\st_cur[1]_i_7_n_0 ),
        .I4(\st_cur[1]_i_8_n_0 ),
        .I5(\st_cur[1]_i_9_n_0 ),
        .O(\st_cur[1]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT5 #(
    .INIT(32'hFFC011FF)) 
    \st_cur[1]_i_3 
       (.I0(prbs_stat[21]),
        .I1(\st_cur[1]_i_10_n_0 ),
        .I2(\u_tester/rx_err0 ),
        .I3(\u_tester/st_cur [0]),
        .I4(\u_tester/st_cur [1]),
        .O(\u_tester/st_nxt ));
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \st_cur[1]_i_4 
       (.I0(\u_tester/mask_count_reg_n_0_[46] ),
        .I1(\u_tester/mask_count_reg_n_0_[47] ),
        .I2(\u_tester/mask_count_reg_n_0_[44] ),
        .I3(\u_tester/mask_count_reg_n_0_[45] ),
        .I4(\st_cur[1]_i_11_n_0 ),
        .O(\st_cur[1]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \st_cur[1]_i_5 
       (.I0(\u_tester/mask_count_reg_n_0_[58] ),
        .I1(\u_tester/mask_count_reg_n_0_[59] ),
        .I2(\u_tester/mask_count_reg_n_0_[56] ),
        .I3(\u_tester/mask_count_reg_n_0_[57] ),
        .I4(\st_cur[1]_i_12_n_0 ),
        .O(\st_cur[1]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \st_cur[1]_i_6 
       (.I0(\u_tester/mask_count_reg_n_0_[26] ),
        .I1(\u_tester/mask_count_reg_n_0_[27] ),
        .I2(\u_tester/mask_count_reg_n_0_[24] ),
        .I3(\u_tester/mask_count_reg_n_0_[25] ),
        .I4(\st_cur[1]_i_13_n_0 ),
        .O(\st_cur[1]_i_6_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \st_cur[1]_i_7 
       (.I0(\u_tester/mask_count_reg_n_0_[38] ),
        .I1(\u_tester/mask_count_reg_n_0_[39] ),
        .I2(\u_tester/mask_count_reg_n_0_[36] ),
        .I3(\u_tester/mask_count_reg_n_0_[37] ),
        .I4(\st_cur[1]_i_14_n_0 ),
        .O(\st_cur[1]_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \st_cur[1]_i_8 
       (.I0(\u_tester/mask_count_reg_n_0_[14] ),
        .I1(\u_tester/mask_count_reg_n_0_[15] ),
        .I2(\u_tester/mask_count_reg_n_0_[12] ),
        .I3(\u_tester/mask_count_reg_n_0_[13] ),
        .I4(\st_cur[1]_i_15_n_0 ),
        .O(\st_cur[1]_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \st_cur[1]_i_9 
       (.I0(\u_tester/mask_count_reg_n_0_[6] ),
        .I1(\u_tester/mask_count_reg_n_0_[7] ),
        .I2(\u_tester/mask_count_reg_n_0_[4] ),
        .I3(\u_tester/mask_count_reg_n_0_[5] ),
        .I4(\st_cur[1]_i_16_n_0 ),
        .O(\st_cur[1]_i_9_n_0 ));
  FDRE \trxiq_rx/data1_reg_reg[0] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data1 [0]),
        .Q(\trxiq_rx/data1_reg [0]),
        .R(1'b0));
  FDRE \trxiq_rx/data1_reg_reg[10] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data1 [10]),
        .Q(\trxiq_rx/data1_reg [10]),
        .R(1'b0));
  FDRE \trxiq_rx/data1_reg_reg[11] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data1 [11]),
        .Q(\trxiq_rx/data1_reg [11]),
        .R(1'b0));
  FDRE \trxiq_rx/data1_reg_reg[1] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data1 [1]),
        .Q(\trxiq_rx/data1_reg [1]),
        .R(1'b0));
  FDRE \trxiq_rx/data1_reg_reg[2] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data1 [2]),
        .Q(\trxiq_rx/data1_reg [2]),
        .R(1'b0));
  FDRE \trxiq_rx/data1_reg_reg[3] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data1 [3]),
        .Q(\trxiq_rx/data1_reg [3]),
        .R(1'b0));
  FDRE \trxiq_rx/data1_reg_reg[4] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data1 [4]),
        .Q(\trxiq_rx/data1_reg [4]),
        .R(1'b0));
  FDRE \trxiq_rx/data1_reg_reg[5] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data1 [5]),
        .Q(\trxiq_rx/data1_reg [5]),
        .R(1'b0));
  FDRE \trxiq_rx/data1_reg_reg[6] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data1 [6]),
        .Q(\trxiq_rx/data1_reg [6]),
        .R(1'b0));
  FDRE \trxiq_rx/data1_reg_reg[7] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data1 [7]),
        .Q(\trxiq_rx/data1_reg [7]),
        .R(1'b0));
  FDRE \trxiq_rx/data1_reg_reg[8] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data1 [8]),
        .Q(\trxiq_rx/data1_reg [8]),
        .R(1'b0));
  FDRE \trxiq_rx/data1_reg_reg[9] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data1 [9]),
        .Q(\trxiq_rx/data1_reg [9]),
        .R(1'b0));
  FDRE \trxiq_rx/data2_reg_reg[0] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data2 [0]),
        .Q(\trxiq_rx/data2_reg [0]),
        .R(1'b0));
  FDRE \trxiq_rx/data2_reg_reg[10] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data2 [10]),
        .Q(\trxiq_rx/data2_reg [10]),
        .R(1'b0));
  FDRE \trxiq_rx/data2_reg_reg[11] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data2 [11]),
        .Q(\trxiq_rx/data2_reg [11]),
        .R(1'b0));
  FDRE \trxiq_rx/data2_reg_reg[1] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data2 [1]),
        .Q(\trxiq_rx/data2_reg [1]),
        .R(1'b0));
  FDRE \trxiq_rx/data2_reg_reg[2] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data2 [2]),
        .Q(\trxiq_rx/data2_reg [2]),
        .R(1'b0));
  FDRE \trxiq_rx/data2_reg_reg[3] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data2 [3]),
        .Q(\trxiq_rx/data2_reg [3]),
        .R(1'b0));
  FDRE \trxiq_rx/data2_reg_reg[4] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data2 [4]),
        .Q(\trxiq_rx/data2_reg [4]),
        .R(1'b0));
  FDRE \trxiq_rx/data2_reg_reg[5] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data2 [5]),
        .Q(\trxiq_rx/data2_reg [5]),
        .R(1'b0));
  FDRE \trxiq_rx/data2_reg_reg[6] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data2 [6]),
        .Q(\trxiq_rx/data2_reg [6]),
        .R(1'b0));
  FDRE \trxiq_rx/data2_reg_reg[7] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data2 [7]),
        .Q(\trxiq_rx/data2_reg [7]),
        .R(1'b0));
  FDRE \trxiq_rx/data2_reg_reg[8] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data2 [8]),
        .Q(\trxiq_rx/data2_reg [8]),
        .R(1'b0));
  FDRE \trxiq_rx/data2_reg_reg[9] 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(\trxiq_rx/data2 [9]),
        .Q(\trxiq_rx/data2_reg [9]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[10] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2_reg [6]),
        .Q(RX_data_a_from_lms[10]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[11] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2_reg [7]),
        .Q(RX_data_a_from_lms[11]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[12] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2_reg [8]),
        .Q(RX_data_a_from_lms[12]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[13] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2_reg [9]),
        .Q(RX_data_a_from_lms[13]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[14] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2_reg [10]),
        .Q(RX_data_a_from_lms[14]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[15] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2_reg [11]),
        .Q(RX_data_a_from_lms[15]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[20] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1_reg [0]),
        .Q(RX_data_a_from_lms[20]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[21] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1_reg [1]),
        .Q(RX_data_a_from_lms[21]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[22] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1_reg [2]),
        .Q(RX_data_a_from_lms[22]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[23] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1_reg [3]),
        .Q(RX_data_a_from_lms[23]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[24] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1_reg [4]),
        .Q(RX_data_a_from_lms[24]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[25] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1_reg [5]),
        .Q(RX_data_a_from_lms[25]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[26] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1_reg [6]),
        .Q(RX_data_a_from_lms[26]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[27] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1_reg [7]),
        .Q(RX_data_a_from_lms[27]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[28] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1_reg [8]),
        .Q(RX_data_a_from_lms[28]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[29] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1_reg [9]),
        .Q(RX_data_a_from_lms[29]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[30] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1_reg [10]),
        .Q(RX_data_a_from_lms[30]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[31] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1_reg [11]),
        .Q(RX_data_a_from_lms[31]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[4] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2_reg [0]),
        .Q(RX_data_a_from_lms[4]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[5] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2_reg [1]),
        .Q(RX_data_a_from_lms[5]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[6] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2_reg [2]),
        .Q(RX_data_a_from_lms[6]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[7] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2_reg [3]),
        .Q(RX_data_a_from_lms[7]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[8] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2_reg [4]),
        .Q(RX_data_a_from_lms[8]),
        .R(1'b0));
  FDRE \trxiq_rx/data_a_reg[9] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2_reg [5]),
        .Q(RX_data_a_from_lms[9]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[10] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2 [6]),
        .Q(RX_data_b_from_lms[10]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[11] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2 [7]),
        .Q(RX_data_b_from_lms[11]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[12] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2 [8]),
        .Q(RX_data_b_from_lms[12]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[13] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2 [9]),
        .Q(RX_data_b_from_lms[13]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[14] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2 [10]),
        .Q(RX_data_b_from_lms[14]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[15] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2 [11]),
        .Q(RX_data_b_from_lms[15]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[20] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1 [0]),
        .Q(RX_data_b_from_lms[20]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[21] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1 [1]),
        .Q(RX_data_b_from_lms[21]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[22] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1 [2]),
        .Q(RX_data_b_from_lms[22]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[23] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1 [3]),
        .Q(RX_data_b_from_lms[23]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[24] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1 [4]),
        .Q(RX_data_b_from_lms[24]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[25] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1 [5]),
        .Q(RX_data_b_from_lms[25]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[26] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1 [6]),
        .Q(RX_data_b_from_lms[26]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[27] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1 [7]),
        .Q(RX_data_b_from_lms[27]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[28] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1 [8]),
        .Q(RX_data_b_from_lms[28]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[29] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1 [9]),
        .Q(RX_data_b_from_lms[29]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[30] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1 [10]),
        .Q(RX_data_b_from_lms[30]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[31] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data1 [11]),
        .Q(RX_data_b_from_lms[31]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[4] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2 [0]),
        .Q(RX_data_b_from_lms[4]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[5] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2 [1]),
        .Q(RX_data_b_from_lms[5]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[6] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2 [2]),
        .Q(RX_data_b_from_lms[6]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[7] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2 [3]),
        .Q(RX_data_b_from_lms[7]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[8] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2 [4]),
        .Q(RX_data_b_from_lms[8]),
        .R(1'b0));
  FDRE \trxiq_rx/data_b_reg[9] 
       (.C(data_clk_2x),
        .CE(\trxiq_rx/iddr_sel_n_0 ),
        .D(\trxiq_rx/data2 [5]),
        .Q(RX_data_b_from_lms[9]),
        .R(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/gen_diq_bits[0].iddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_D[0]),
        .Q1(\trxiq_rx/data1 [0]),
        .Q2(\trxiq_rx/data2 [0]),
        .R(1'b0),
        .S(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/gen_diq_bits[10].iddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_D[10]),
        .Q1(\trxiq_rx/data1 [10]),
        .Q2(\trxiq_rx/data2 [10]),
        .R(1'b0),
        .S(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/gen_diq_bits[11].iddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_D[11]),
        .Q1(\trxiq_rx/data1 [11]),
        .Q2(\trxiq_rx/data2 [11]),
        .R(1'b0),
        .S(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/gen_diq_bits[1].iddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_D[1]),
        .Q1(\trxiq_rx/data1 [1]),
        .Q2(\trxiq_rx/data2 [1]),
        .R(1'b0),
        .S(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/gen_diq_bits[2].iddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_D[2]),
        .Q1(\trxiq_rx/data1 [2]),
        .Q2(\trxiq_rx/data2 [2]),
        .R(1'b0),
        .S(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/gen_diq_bits[3].iddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_D[3]),
        .Q1(\trxiq_rx/data1 [3]),
        .Q2(\trxiq_rx/data2 [3]),
        .R(1'b0),
        .S(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/gen_diq_bits[4].iddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_D[4]),
        .Q1(\trxiq_rx/data1 [4]),
        .Q2(\trxiq_rx/data2 [4]),
        .R(1'b0),
        .S(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/gen_diq_bits[5].iddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_D[5]),
        .Q1(\trxiq_rx/data1 [5]),
        .Q2(\trxiq_rx/data2 [5]),
        .R(1'b0),
        .S(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/gen_diq_bits[6].iddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_D[6]),
        .Q1(\trxiq_rx/data1 [6]),
        .Q2(\trxiq_rx/data2 [6]),
        .R(1'b0),
        .S(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/gen_diq_bits[7].iddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_D[7]),
        .Q1(\trxiq_rx/data1 [7]),
        .Q2(\trxiq_rx/data2 [7]),
        .R(1'b0),
        .S(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/gen_diq_bits[8].iddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_D[8]),
        .Q1(\trxiq_rx/data1 [8]),
        .Q2(\trxiq_rx/data2 [8]),
        .R(1'b0),
        .S(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/gen_diq_bits[9].iddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_D[9]),
        .Q1(\trxiq_rx/data1 [9]),
        .Q2(\trxiq_rx/data2 [9]),
        .R(1'b0),
        .S(1'b0));
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  IDDR #(
    .DDR_CLK_EDGE("SAME_EDGE_PIPELINED"),
    .INIT_Q1(1'b0),
    .INIT_Q2(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_rx/iddr_sel 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(LMS_DIQ2_IQSEL),
        .Q1(\trxiq_rx/iddr_sel_n_0 ),
        .Q2(\NLW_trxiq_rx/iddr_sel_Q2_UNCONNECTED ),
        .R(1'b0),
        .S(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[10] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[10]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[10] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[11] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[11]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[11] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[12] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[12]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[12] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[13] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[13]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[13] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[14] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[14]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[14] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[15] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[15]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[15] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[20] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[20]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[20] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[21] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[21]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[21] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[22] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[22]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[22] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[23] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[23]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[23] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[24] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[24]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[24] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[25] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[25]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[25] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[26] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[26]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[26] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[27] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[27]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[27] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[28] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[28]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[28] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[29] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[29]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[29] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[30] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[30]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[30] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[31] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[31]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[31] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[4] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[4]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[4] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[5] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[5]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[5] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[6] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[6]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[6] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[7] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[7]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[7] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[8] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[8]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[8] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_a_reg_reg[9] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_a_to_lms[9]),
        .Q(\trxiq_tx/data_a_reg_reg_n_0_[9] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[10] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[10]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[10] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[11] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[11]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[11] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[12] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[12]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[12] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[13] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[13]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[13] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[14] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[14]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[14] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[15] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[15]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[15] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[20] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[20]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[20] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[21] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[21]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[21] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[22] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[22]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[22] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[23] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[23]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[23] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[24] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[24]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[24] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[25] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[25]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[25] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[26] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[26]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[26] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[27] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[27]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[27] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[28] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[28]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[28] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[29] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[29]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[29] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[30] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[30]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[30] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[31] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[31]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[31] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[4] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[4]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[4] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[5] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[5]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[5] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[6] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[6]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[6] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[7] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[7]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[7] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[8] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[8]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[8] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_b_reg_reg[9] 
       (.C(data_clk_2x),
        .CE(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D(TX_data_b_to_lms[9]),
        .Q(\trxiq_tx/data_b_reg_reg_n_0_[9] ),
        .R(1'b0));
  FDRE \trxiq_tx/data_stb_i_reg 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D(data_stb_i_i_1_n_0),
        .Q(\trxiq_tx/data_stb_i_reg_n_0 ),
        .R(1'b0));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/gen_diq_bits[0].oddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/gen_diq_bits[0].oddr_data_i_1_n_0 ),
        .D2(\trxiq_tx/gen_diq_bits[0].oddr_data_i_2_n_0 ),
        .Q(LMS_DIQ1_D[0]),
        .R(\NLW_trxiq_tx/gen_diq_bits[0].oddr_data_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/gen_diq_bits[0].oddr_data_S_UNCONNECTED ));
  (* SOFT_HLUTNM = "soft_lutpair40" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[0].oddr_data_i_1 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[20] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[20] ),
        .O(\trxiq_tx/gen_diq_bits[0].oddr_data_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair52" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[0].oddr_data_i_2 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[4] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[4] ),
        .O(\trxiq_tx/gen_diq_bits[0].oddr_data_i_2_n_0 ));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/gen_diq_bits[10].oddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/gen_diq_bits[10].oddr_data_i_1_n_0 ),
        .D2(\trxiq_tx/gen_diq_bits[10].oddr_data_i_2_n_0 ),
        .Q(LMS_DIQ1_D[10]),
        .R(\NLW_trxiq_tx/gen_diq_bits[10].oddr_data_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/gen_diq_bits[10].oddr_data_S_UNCONNECTED ));
  (* SOFT_HLUTNM = "soft_lutpair51" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[10].oddr_data_i_1 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[30] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[30] ),
        .O(\trxiq_tx/gen_diq_bits[10].oddr_data_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair44" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[10].oddr_data_i_2 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[14] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[14] ),
        .O(\trxiq_tx/gen_diq_bits[10].oddr_data_i_2_n_0 ));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/gen_diq_bits[11].oddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/D1 ),
        .D2(\trxiq_tx/D2 ),
        .Q(LMS_DIQ1_D[11]),
        .R(\NLW_trxiq_tx/gen_diq_bits[11].oddr_data_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/gen_diq_bits[11].oddr_data_S_UNCONNECTED ));
  (* SOFT_HLUTNM = "soft_lutpair51" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[11].oddr_data_i_1 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[31] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[31] ),
        .O(\trxiq_tx/D1 ));
  (* SOFT_HLUTNM = "soft_lutpair45" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[11].oddr_data_i_2 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[15] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[15] ),
        .O(\trxiq_tx/D2 ));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/gen_diq_bits[1].oddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/gen_diq_bits[1].oddr_data_i_1_n_0 ),
        .D2(\trxiq_tx/gen_diq_bits[1].oddr_data_i_2_n_0 ),
        .Q(LMS_DIQ1_D[1]),
        .R(\NLW_trxiq_tx/gen_diq_bits[1].oddr_data_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/gen_diq_bits[1].oddr_data_S_UNCONNECTED ));
  (* SOFT_HLUTNM = "soft_lutpair42" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[1].oddr_data_i_1 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[21] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[21] ),
        .O(\trxiq_tx/gen_diq_bits[1].oddr_data_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair52" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[1].oddr_data_i_2 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[5] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[5] ),
        .O(\trxiq_tx/gen_diq_bits[1].oddr_data_i_2_n_0 ));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/gen_diq_bits[2].oddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/gen_diq_bits[2].oddr_data_i_1_n_0 ),
        .D2(\trxiq_tx/gen_diq_bits[2].oddr_data_i_2_n_0 ),
        .Q(LMS_DIQ1_D[2]),
        .R(\NLW_trxiq_tx/gen_diq_bits[2].oddr_data_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/gen_diq_bits[2].oddr_data_S_UNCONNECTED ));
  (* SOFT_HLUTNM = "soft_lutpair44" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[2].oddr_data_i_1 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[22] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[22] ),
        .O(\trxiq_tx/gen_diq_bits[2].oddr_data_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair53" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[2].oddr_data_i_2 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[6] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[6] ),
        .O(\trxiq_tx/gen_diq_bits[2].oddr_data_i_2_n_0 ));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/gen_diq_bits[3].oddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/gen_diq_bits[3].oddr_data_i_1_n_0 ),
        .D2(\trxiq_tx/gen_diq_bits[3].oddr_data_i_2_n_0 ),
        .Q(LMS_DIQ1_D[3]),
        .R(\NLW_trxiq_tx/gen_diq_bits[3].oddr_data_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/gen_diq_bits[3].oddr_data_S_UNCONNECTED ));
  (* SOFT_HLUTNM = "soft_lutpair45" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[3].oddr_data_i_1 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[23] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[23] ),
        .O(\trxiq_tx/gen_diq_bits[3].oddr_data_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair53" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[3].oddr_data_i_2 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[7] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[7] ),
        .O(\trxiq_tx/gen_diq_bits[3].oddr_data_i_2_n_0 ));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/gen_diq_bits[4].oddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/gen_diq_bits[4].oddr_data_i_1_n_0 ),
        .D2(\trxiq_tx/gen_diq_bits[4].oddr_data_i_2_n_0 ),
        .Q(LMS_DIQ1_D[4]),
        .R(\NLW_trxiq_tx/gen_diq_bits[4].oddr_data_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/gen_diq_bits[4].oddr_data_S_UNCONNECTED ));
  (* SOFT_HLUTNM = "soft_lutpair48" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[4].oddr_data_i_1 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[24] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[24] ),
        .O(\trxiq_tx/gen_diq_bits[4].oddr_data_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair40" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[4].oddr_data_i_2 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[8] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[8] ),
        .O(\trxiq_tx/gen_diq_bits[4].oddr_data_i_2_n_0 ));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/gen_diq_bits[5].oddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/gen_diq_bits[5].oddr_data_i_1_n_0 ),
        .D2(\trxiq_tx/gen_diq_bits[5].oddr_data_i_2_n_0 ),
        .Q(LMS_DIQ1_D[5]),
        .R(\NLW_trxiq_tx/gen_diq_bits[5].oddr_data_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/gen_diq_bits[5].oddr_data_S_UNCONNECTED ));
  (* SOFT_HLUTNM = "soft_lutpair48" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[5].oddr_data_i_1 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[25] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[25] ),
        .O(\trxiq_tx/gen_diq_bits[5].oddr_data_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair54" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[5].oddr_data_i_2 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[9] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[9] ),
        .O(\trxiq_tx/gen_diq_bits[5].oddr_data_i_2_n_0 ));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/gen_diq_bits[6].oddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/gen_diq_bits[6].oddr_data_i_1_n_0 ),
        .D2(\trxiq_tx/gen_diq_bits[6].oddr_data_i_2_n_0 ),
        .Q(LMS_DIQ1_D[6]),
        .R(\NLW_trxiq_tx/gen_diq_bits[6].oddr_data_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/gen_diq_bits[6].oddr_data_S_UNCONNECTED ));
  (* SOFT_HLUTNM = "soft_lutpair49" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[6].oddr_data_i_1 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[26] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[26] ),
        .O(\trxiq_tx/gen_diq_bits[6].oddr_data_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair54" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[6].oddr_data_i_2 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[10] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[10] ),
        .O(\trxiq_tx/gen_diq_bits[6].oddr_data_i_2_n_0 ));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/gen_diq_bits[7].oddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/gen_diq_bits[7].oddr_data_i_1_n_0 ),
        .D2(\trxiq_tx/gen_diq_bits[7].oddr_data_i_2_n_0 ),
        .Q(LMS_DIQ1_D[7]),
        .R(\NLW_trxiq_tx/gen_diq_bits[7].oddr_data_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/gen_diq_bits[7].oddr_data_S_UNCONNECTED ));
  (* SOFT_HLUTNM = "soft_lutpair49" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[7].oddr_data_i_1 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[27] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[27] ),
        .O(\trxiq_tx/gen_diq_bits[7].oddr_data_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair55" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[7].oddr_data_i_2 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[11] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[11] ),
        .O(\trxiq_tx/gen_diq_bits[7].oddr_data_i_2_n_0 ));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/gen_diq_bits[8].oddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/gen_diq_bits[8].oddr_data_i_1_n_0 ),
        .D2(\trxiq_tx/gen_diq_bits[8].oddr_data_i_2_n_0 ),
        .Q(LMS_DIQ1_D[8]),
        .R(\NLW_trxiq_tx/gen_diq_bits[8].oddr_data_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/gen_diq_bits[8].oddr_data_S_UNCONNECTED ));
  (* SOFT_HLUTNM = "soft_lutpair50" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[8].oddr_data_i_1 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[28] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[28] ),
        .O(\trxiq_tx/gen_diq_bits[8].oddr_data_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair55" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[8].oddr_data_i_2 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[12] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[12] ),
        .O(\trxiq_tx/gen_diq_bits[8].oddr_data_i_2_n_0 ));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/gen_diq_bits[9].oddr_data 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/gen_diq_bits[9].oddr_data_i_1_n_0 ),
        .D2(\trxiq_tx/gen_diq_bits[9].oddr_data_i_2_n_0 ),
        .Q(LMS_DIQ1_D[9]),
        .R(\NLW_trxiq_tx/gen_diq_bits[9].oddr_data_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/gen_diq_bits[9].oddr_data_S_UNCONNECTED ));
  (* SOFT_HLUTNM = "soft_lutpair50" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[9].oddr_data_i_1 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[29] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[29] ),
        .O(\trxiq_tx/gen_diq_bits[9].oddr_data_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair42" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \trxiq_tx/gen_diq_bits[9].oddr_data_i_2 
       (.I0(\trxiq_tx/data_b_reg_reg_n_0_[13] ),
        .I1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .I2(\trxiq_tx/data_a_reg_reg_n_0_[13] ),
        .O(\trxiq_tx/gen_diq_bits[9].oddr_data_i_2_n_0 ));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("SAME_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \trxiq_tx/oddr_sel 
       (.C(data_clk_2x),
        .CE(1'b1),
        .D1(\trxiq_tx/data_stb_i_reg_n_0 ),
        .D2(\trxiq_tx/data_stb_i_reg_n_0 ),
        .Q(LMS_DIQ1_IQSEL),
        .R(\NLW_trxiq_tx/oddr_sel_R_UNCONNECTED ),
        .S(\NLW_trxiq_tx/oddr_sel_S_UNCONNECTED ));
  LUT3 #(
    .INIT(8'h80)) 
    \tx_cnt[0]_i_1 
       (.I0(prbs_ctrl[12]),
        .I1(\u_tester/st_cur [1]),
        .I2(\u_tester/st_cur [0]),
        .O(\u_tester/tx_cnt0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \tx_cnt[0]_i_3 
       (.I0(\u_tester/tx_cnt_reg_n_0_[0] ),
        .O(\tx_cnt[0]_i_3_n_0 ));
  CARRY4 \tx_cnt_reg[0]_i_2 
       (.CI(1'b0),
        .CO({\tx_cnt_reg[0]_i_2_n_0 ,\tx_cnt_reg[0]_i_2_n_1 ,\tx_cnt_reg[0]_i_2_n_2 ,\tx_cnt_reg[0]_i_2_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b1}),
        .O({\tx_cnt_reg[0]_i_2_n_4 ,\tx_cnt_reg[0]_i_2_n_5 ,\tx_cnt_reg[0]_i_2_n_6 ,\tx_cnt_reg[0]_i_2_n_7 }),
        .S({\u_tester/tx_cnt_reg_n_0_[3] ,\u_tester/tx_cnt_reg_n_0_[2] ,\u_tester/tx_cnt_reg_n_0_[1] ,\tx_cnt[0]_i_3_n_0 }));
  CARRY4 \tx_cnt_reg[12]_i_1 
       (.CI(\tx_cnt_reg[8]_i_1_n_0 ),
        .CO({\tx_cnt_reg[12]_i_1_n_0 ,\tx_cnt_reg[12]_i_1_n_1 ,\tx_cnt_reg[12]_i_1_n_2 ,\tx_cnt_reg[12]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\tx_cnt_reg[12]_i_1_n_4 ,\tx_cnt_reg[12]_i_1_n_5 ,\tx_cnt_reg[12]_i_1_n_6 ,\tx_cnt_reg[12]_i_1_n_7 }),
        .S({\u_tester/tx_cnt_reg_n_0_[15] ,\u_tester/tx_cnt_reg_n_0_[14] ,\u_tester/tx_cnt_reg_n_0_[13] ,\u_tester/tx_cnt_reg_n_0_[12] }));
  CARRY4 \tx_cnt_reg[16]_i_1 
       (.CI(\tx_cnt_reg[12]_i_1_n_0 ),
        .CO({\tx_cnt_reg[16]_i_1_n_0 ,\tx_cnt_reg[16]_i_1_n_1 ,\tx_cnt_reg[16]_i_1_n_2 ,\tx_cnt_reg[16]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\tx_cnt_reg[16]_i_1_n_4 ,\tx_cnt_reg[16]_i_1_n_5 ,\tx_cnt_reg[16]_i_1_n_6 ,\tx_cnt_reg[16]_i_1_n_7 }),
        .S({\u_tester/tx_cnt_reg_n_0_[19] ,\u_tester/tx_cnt_reg_n_0_[18] ,\u_tester/tx_cnt_reg_n_0_[17] ,\u_tester/tx_cnt_reg_n_0_[16] }));
  CARRY4 \tx_cnt_reg[20]_i_1 
       (.CI(\tx_cnt_reg[16]_i_1_n_0 ),
        .CO({\tx_cnt_reg[20]_i_1_n_0 ,\tx_cnt_reg[20]_i_1_n_1 ,\tx_cnt_reg[20]_i_1_n_2 ,\tx_cnt_reg[20]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\tx_cnt_reg[20]_i_1_n_4 ,\tx_cnt_reg[20]_i_1_n_5 ,\tx_cnt_reg[20]_i_1_n_6 ,\tx_cnt_reg[20]_i_1_n_7 }),
        .S({prbs_stat[8],\u_tester/tx_cnt_reg_n_0_[22] ,\u_tester/tx_cnt_reg_n_0_[21] ,\u_tester/tx_cnt_reg_n_0_[20] }));
  CARRY4 \tx_cnt_reg[24]_i_1 
       (.CI(\tx_cnt_reg[20]_i_1_n_0 ),
        .CO({\tx_cnt_reg[24]_i_1_n_0 ,\tx_cnt_reg[24]_i_1_n_1 ,\tx_cnt_reg[24]_i_1_n_2 ,\tx_cnt_reg[24]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\tx_cnt_reg[24]_i_1_n_4 ,\tx_cnt_reg[24]_i_1_n_5 ,\tx_cnt_reg[24]_i_1_n_6 ,\tx_cnt_reg[24]_i_1_n_7 }),
        .S(prbs_stat[12:9]));
  CARRY4 \tx_cnt_reg[28]_i_1 
       (.CI(\tx_cnt_reg[24]_i_1_n_0 ),
        .CO({\NLW_tx_cnt_reg[28]_i_1_CO_UNCONNECTED [3:2],\tx_cnt_reg[28]_i_1_n_2 ,\tx_cnt_reg[28]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\NLW_tx_cnt_reg[28]_i_1_O_UNCONNECTED [3],\tx_cnt_reg[28]_i_1_n_5 ,\tx_cnt_reg[28]_i_1_n_6 ,\tx_cnt_reg[28]_i_1_n_7 }),
        .S({1'b0,prbs_stat[15:13]}));
  CARRY4 \tx_cnt_reg[4]_i_1 
       (.CI(\tx_cnt_reg[0]_i_2_n_0 ),
        .CO({\tx_cnt_reg[4]_i_1_n_0 ,\tx_cnt_reg[4]_i_1_n_1 ,\tx_cnt_reg[4]_i_1_n_2 ,\tx_cnt_reg[4]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\tx_cnt_reg[4]_i_1_n_4 ,\tx_cnt_reg[4]_i_1_n_5 ,\tx_cnt_reg[4]_i_1_n_6 ,\tx_cnt_reg[4]_i_1_n_7 }),
        .S({\u_tester/tx_cnt_reg_n_0_[7] ,\u_tester/tx_cnt_reg_n_0_[6] ,\u_tester/tx_cnt_reg_n_0_[5] ,\u_tester/tx_cnt_reg_n_0_[4] }));
  CARRY4 \tx_cnt_reg[8]_i_1 
       (.CI(\tx_cnt_reg[4]_i_1_n_0 ),
        .CO({\tx_cnt_reg[8]_i_1_n_0 ,\tx_cnt_reg[8]_i_1_n_1 ,\tx_cnt_reg[8]_i_1_n_2 ,\tx_cnt_reg[8]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\tx_cnt_reg[8]_i_1_n_4 ,\tx_cnt_reg[8]_i_1_n_5 ,\tx_cnt_reg[8]_i_1_n_6 ,\tx_cnt_reg[8]_i_1_n_7 }),
        .S({\u_tester/tx_cnt_reg_n_0_[11] ,\u_tester/tx_cnt_reg_n_0_[10] ,\u_tester/tx_cnt_reg_n_0_[9] ,\u_tester/tx_cnt_reg_n_0_[8] }));
  FDRE \tx_cycle_counter/counter_src_reg[0] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[0]_i_1_n_7 ),
        .Q(\tx_cycle_counter/counter_src_reg [0]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[10] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[8]_i_1_n_5 ),
        .Q(\tx_cycle_counter/counter_src_reg [10]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[11] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[8]_i_1_n_4 ),
        .Q(\tx_cycle_counter/counter_src_reg [11]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[12] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[12]_i_1_n_7 ),
        .Q(\tx_cycle_counter/counter_src_reg [12]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[13] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[12]_i_1_n_6 ),
        .Q(\tx_cycle_counter/counter_src_reg [13]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[14] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[12]_i_1_n_5 ),
        .Q(\tx_cycle_counter/counter_src_reg [14]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[15] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[12]_i_1_n_4 ),
        .Q(\tx_cycle_counter/counter_src_reg [15]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[16] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[16]_i_1_n_7 ),
        .Q(\tx_cycle_counter/counter_src_reg [16]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[17] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[16]_i_1_n_6 ),
        .Q(\tx_cycle_counter/counter_src_reg [17]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[18] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[16]_i_1_n_5 ),
        .Q(\tx_cycle_counter/counter_src_reg [18]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[19] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[16]_i_1_n_4 ),
        .Q(\tx_cycle_counter/counter_src_reg [19]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[1] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[0]_i_1_n_6 ),
        .Q(\tx_cycle_counter/counter_src_reg [1]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[20] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[20]_i_1_n_7 ),
        .Q(\tx_cycle_counter/counter_src_reg [20]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[21] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[20]_i_1_n_6 ),
        .Q(\tx_cycle_counter/counter_src_reg [21]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[22] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[20]_i_1_n_5 ),
        .Q(\tx_cycle_counter/counter_src_reg [22]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[23] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[20]_i_1_n_4 ),
        .Q(\tx_cycle_counter/counter_src_reg [23]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[24] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[24]_i_1_n_7 ),
        .Q(\tx_cycle_counter/counter_src_reg [24]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[25] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[24]_i_1_n_6 ),
        .Q(\tx_cycle_counter/counter_src_reg [25]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[26] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[24]_i_1_n_5 ),
        .Q(\tx_cycle_counter/counter_src_reg [26]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[27] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[24]_i_1_n_4 ),
        .Q(\tx_cycle_counter/counter_src_reg [27]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[28] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[28]_i_1_n_7 ),
        .Q(\tx_cycle_counter/counter_src_reg [28]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[29] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[28]_i_1_n_6 ),
        .Q(\tx_cycle_counter/counter_src_reg [29]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[2] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[0]_i_1_n_5 ),
        .Q(\tx_cycle_counter/counter_src_reg [2]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[30] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[28]_i_1_n_5 ),
        .Q(\tx_cycle_counter/counter_src_reg [30]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[31] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[28]_i_1_n_4 ),
        .Q(\tx_cycle_counter/counter_src_reg [31]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[3] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[0]_i_1_n_4 ),
        .Q(\tx_cycle_counter/counter_src_reg [3]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[4] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[4]_i_1_n_7 ),
        .Q(\tx_cycle_counter/counter_src_reg [4]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[5] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[4]_i_1_n_6 ),
        .Q(\tx_cycle_counter/counter_src_reg [5]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[6] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[4]_i_1_n_5 ),
        .Q(\tx_cycle_counter/counter_src_reg [6]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[7] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[4]_i_1_n_4 ),
        .Q(\tx_cycle_counter/counter_src_reg [7]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[8] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[8]_i_1_n_7 ),
        .Q(\tx_cycle_counter/counter_src_reg [8]),
        .R(DATA_rst));
  FDRE \tx_cycle_counter/counter_src_reg[9] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\counter_src_reg[8]_i_1_n_6 ),
        .Q(\tx_cycle_counter/counter_src_reg [9]),
        .R(DATA_rst));
  (* DEST_SYNC_FF = "2" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* REG_OUTPUT = "1" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SIM_LOSSLESS_GRAY_CHK = "0" *) 
  (* VERSION = "0" *) 
  (* WIDTH = "32" *) 
  (* XPM_CDC = "GRAY" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_gray \tx_cycle_counter/u_xfer 
       (.dest_clk(SYS_clk),
        .dest_out_bin(SYS_data_clk_counter),
        .src_clk(DATA_clk),
        .src_in_bin(\tx_cycle_counter/counter_src_reg ));
  (* DEST_SYNC_FF = "4" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_INPUT_REG = "1" *) 
  (* VERSION = "0" *) 
  (* WIDTH = "16" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_array_single__parameterized3 tx_prbs_ctrl_xfer
       (.dest_clk(DATA_clk),
        .dest_out(prbs_ctrl),
        .src_clk(SYS_clk),
        .src_in(SYS_prbs_ctrl));
  (* DEST_SYNC_FF = "4" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_INPUT_REG = "1" *) 
  (* VERSION = "0" *) 
  (* WIDTH = "96" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_array_single__parameterized2 tx_prbs_stat_xfer
       (.dest_clk(SYS_clk),
        .dest_out({SYS_prbs_stat,SYS_prbs_e}),
        .src_clk(DATA_clk),
        .src_in({prbs_ctrl[11],prbs_ctrl[11],prbs_ctrl[15],prbs_ctrl[12],prbs_stat[27:24],prbs_ctrl[13],prbs_ctrl[14],prbs_stat[21:0],prbs_e[63:52],1'b0,1'b0,1'b0,1'b0,prbs_e[47:36],1'b0,1'b0,1'b0,1'b0,prbs_e[31:20],1'b0,1'b0,1'b0,1'b0,prbs_e[15:4],1'b0,1'b0,1'b0,1'b0}));
  LUT2 #(
    .INIT(4'h8)) 
    tx_prbs_stat_xfer_i_1
       (.I0(\u_tester/st_cur [0]),
        .I1(\u_tester/st_cur [1]),
        .O(prbs_stat[27]));
  LUT2 #(
    .INIT(4'h2)) 
    tx_prbs_stat_xfer_i_2
       (.I0(\u_tester/st_cur [0]),
        .I1(\u_tester/st_cur [1]),
        .O(prbs_stat[26]));
  (* SOFT_HLUTNM = "soft_lutpair57" *) 
  LUT2 #(
    .INIT(4'h2)) 
    tx_prbs_stat_xfer_i_3
       (.I0(\u_tester/st_cur [1]),
        .I1(\u_tester/st_cur [0]),
        .O(prbs_stat[25]));
  (* SOFT_HLUTNM = "soft_lutpair57" *) 
  LUT2 #(
    .INIT(4'h1)) 
    tx_prbs_stat_xfer_i_4
       (.I0(\u_tester/st_cur [1]),
        .I1(\u_tester/st_cur [0]),
        .O(prbs_stat[24]));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT3 #(
    .INIT(8'hAB)) 
    tx_prbs_stat_xfer_i_5
       (.I0(\u_tester/rx_err0 ),
        .I1(\u_tester/st_cur [1]),
        .I2(\u_tester/st_cur [0]),
        .O(prbs_stat[20]));
  (* DEST_SYNC_FF = "4" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_INPUT_REG = "1" *) 
  (* VERSION = "0" *) 
  (* WIDTH = "64" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_array_single__parameterized1__2 tx_test_data_xfer
       (.dest_clk(DATA_clk),
        .dest_out({TX_test_data_b,tx_test_data_xfer_n_32,tx_test_data_xfer_n_33,tx_test_data_xfer_n_34,tx_test_data_xfer_n_35,tx_test_data_xfer_n_36,tx_test_data_xfer_n_37,tx_test_data_xfer_n_38,tx_test_data_xfer_n_39,tx_test_data_xfer_n_40,tx_test_data_xfer_n_41,tx_test_data_xfer_n_42,tx_test_data_xfer_n_43,tx_test_data_xfer_n_44,tx_test_data_xfer_n_45,tx_test_data_xfer_n_46,tx_test_data_xfer_n_47,tx_test_data_xfer_n_48,tx_test_data_xfer_n_49,tx_test_data_xfer_n_50,tx_test_data_xfer_n_51,tx_test_data_xfer_n_52,tx_test_data_xfer_n_53,tx_test_data_xfer_n_54,tx_test_data_xfer_n_55,tx_test_data_xfer_n_56,tx_test_data_xfer_n_57,tx_test_data_xfer_n_58,tx_test_data_xfer_n_59,tx_test_data_xfer_n_60,tx_test_data_xfer_n_61,tx_test_data_xfer_n_62,tx_test_data_xfer_n_63}),
        .src_clk(SYS_clk),
        .src_in({SYS_test_data_b_tx,SYS_test_data_a_tx}));
  (* DEST_SYNC_FF = "4" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_INPUT_REG = "1" *) 
  (* VERSION = "0" *) 
  (* WIDTH = "3" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_array_single__parameterized0 tx_test_mode_xfer
       (.dest_clk(DATA_clk),
        .dest_out({PRBS_test_signal,tx_test_mode_xfer_n_1,tx_test_mode_xfer_n_2}),
        .src_clk(SYS_clk),
        .src_in(SYS_test_mode[4:2]));
  FDRE \u_clocks/clk_hist_reg[0] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(LMS_DIQ2_MCLK),
        .Q(\u_clocks/clk_hist [0]),
        .R(1'b0));
  FDRE \u_clocks/clk_hist_reg[1] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(\u_clocks/clk_hist [0]),
        .Q(\u_clocks/clk_hist [1]),
        .R(1'b0));
  (* XILINX_LEGACY_PRIM = "BUFGMUX" *) 
  (* XILINX_TRANSFORM_PINMAP = "S:CE1,CE0" *) 
  (* box_type = "PRIMITIVE" *) 
  BUFGCTRL #(
    .INIT_OUT(0),
    .IS_CE0_INVERTED(1'b1),
    .PRESELECT_I0("TRUE"),
    .PRESELECT_I1("FALSE")) 
    \u_clocks/clkmux_1x 
       (.CE0(sel_mmcm),
        .CE1(sel_mmcm),
        .I0(\u_clocks/samp_clk_1x ),
        .I1(\u_clocks/mmcm_clk_1x ),
        .IGNORE0(1'b0),
        .IGNORE1(1'b0),
        .O(DATA_clk),
        .S0(1'b1),
        .S1(1'b1));
  (* XILINX_LEGACY_PRIM = "BUFGMUX" *) 
  (* XILINX_TRANSFORM_PINMAP = "S:CE1,CE0" *) 
  (* box_type = "PRIMITIVE" *) 
  BUFGCTRL #(
    .INIT_OUT(0),
    .IS_CE0_INVERTED(1'b1),
    .PRESELECT_I0("TRUE"),
    .PRESELECT_I1("FALSE")) 
    \u_clocks/clkmux_2x 
       (.CE0(sel_mmcm),
        .CE1(sel_mmcm),
        .I0(\u_clocks/samp_clk_2x ),
        .I1(\u_clocks/mmcm_clk_2x ),
        .IGNORE0(1'b0),
        .IGNORE1(1'b0),
        .O(data_clk_2x),
        .S0(1'b1),
        .S1(1'b1));
  (* XILINX_LEGACY_PRIM = "BUFGMUX" *) 
  (* XILINX_TRANSFORM_PINMAP = "S:CE1,CE0" *) 
  (* box_type = "PRIMITIVE" *) 
  BUFGCTRL #(
    .INIT_OUT(0),
    .IS_CE0_INVERTED(1'b1),
    .PRESELECT_I0("TRUE"),
    .PRESELECT_I1("FALSE")) 
    \u_clocks/clkmux_fb 
       (.CE0(sel_mmcm),
        .CE1(sel_mmcm),
        .I0(\u_clocks/samp_clk_fb_reg_n_0 ),
        .I1(\u_clocks/mmcm_clk_fb ),
        .IGNORE0(1'b0),
        .IGNORE1(1'b0),
        .O(\u_clocks/data_clk_fb ),
        .S0(1'b1),
        .S1(1'b1));
  FDRE \u_clocks/counterFMax_reg[2] 
       (.C(SYS_clk),
        .CE(\counterFMax[7]_i_1_n_0 ),
        .D(counterF[2]),
        .Q(counterFMax[2]),
        .R(1'b0));
  FDRE \u_clocks/counterFMax_reg[3] 
       (.C(SYS_clk),
        .CE(\counterFMax[7]_i_1_n_0 ),
        .D(counterF[3]),
        .Q(counterFMax[3]),
        .R(1'b0));
  FDRE \u_clocks/counterFMax_reg[4] 
       (.C(SYS_clk),
        .CE(\counterFMax[7]_i_1_n_0 ),
        .D(counterF[4]),
        .Q(counterFMax[4]),
        .R(1'b0));
  FDRE \u_clocks/counterFMax_reg[5] 
       (.C(SYS_clk),
        .CE(\counterFMax[7]_i_1_n_0 ),
        .D(counterF[5]),
        .Q(counterFMax[5]),
        .R(1'b0));
  FDRE \u_clocks/counterFMax_reg[6] 
       (.C(SYS_clk),
        .CE(\counterFMax[7]_i_1_n_0 ),
        .D(counterF[6]),
        .Q(counterFMax[6]),
        .R(1'b0));
  FDRE \u_clocks/counterFMax_reg[7] 
       (.C(SYS_clk),
        .CE(\counterFMax[7]_i_1_n_0 ),
        .D(counterF[7]),
        .Q(counterFMax[7]),
        .R(1'b0));
  FDRE \u_clocks/counterF_reg[0] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(\counterF[0]_i_1_n_0 ),
        .Q(counterF[0]),
        .R(\counterFMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterF_reg[1] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(\counterF[1]_i_1_n_0 ),
        .Q(counterF[1]),
        .R(\counterFMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterF_reg[2] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(\counterF[2]_i_1_n_0 ),
        .Q(counterF[2]),
        .R(\counterFMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterF_reg[3] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(\counterF[3]_i_1_n_0 ),
        .Q(counterF[3]),
        .R(\counterFMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterF_reg[4] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(\counterF[4]_i_1_n_0 ),
        .Q(counterF[4]),
        .R(\counterFMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterF_reg[5] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(\counterF[5]_i_1_n_0 ),
        .Q(counterF[5]),
        .R(\counterFMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterF_reg[6] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(\counterF[6]_i_1_n_0 ),
        .Q(counterF[6]),
        .R(\counterFMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterF_reg[7] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(\counterF[7]_i_1_n_0 ),
        .Q(counterF[7]),
        .R(\counterFMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterRMax_reg[2] 
       (.C(SYS_clk),
        .CE(\counterRMax[7]_i_1_n_0 ),
        .D(counterR[2]),
        .Q(counterRMax[2]),
        .R(1'b0));
  FDRE \u_clocks/counterRMax_reg[3] 
       (.C(SYS_clk),
        .CE(\counterRMax[7]_i_1_n_0 ),
        .D(counterR[3]),
        .Q(counterRMax[3]),
        .R(1'b0));
  FDRE \u_clocks/counterRMax_reg[4] 
       (.C(SYS_clk),
        .CE(\counterRMax[7]_i_1_n_0 ),
        .D(counterR[4]),
        .Q(counterRMax[4]),
        .R(1'b0));
  FDRE \u_clocks/counterRMax_reg[5] 
       (.C(SYS_clk),
        .CE(\counterRMax[7]_i_1_n_0 ),
        .D(counterR[5]),
        .Q(counterRMax[5]),
        .R(1'b0));
  FDRE \u_clocks/counterRMax_reg[6] 
       (.C(SYS_clk),
        .CE(\counterRMax[7]_i_1_n_0 ),
        .D(counterR[6]),
        .Q(counterRMax[6]),
        .R(1'b0));
  FDRE \u_clocks/counterRMax_reg[7] 
       (.C(SYS_clk),
        .CE(\counterRMax[7]_i_1_n_0 ),
        .D(counterR[7]),
        .Q(counterRMax[7]),
        .R(1'b0));
  FDRE \u_clocks/counterR_reg[0] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(plusOp[0]),
        .Q(counterR[0]),
        .R(\counterRMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterR_reg[1] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(plusOp[1]),
        .Q(counterR[1]),
        .R(\counterRMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterR_reg[2] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(plusOp[2]),
        .Q(counterR[2]),
        .R(\counterRMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterR_reg[3] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(plusOp[3]),
        .Q(counterR[3]),
        .R(\counterRMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterR_reg[4] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(plusOp[4]),
        .Q(counterR[4]),
        .R(\counterRMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterR_reg[5] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(plusOp[5]),
        .Q(counterR[5]),
        .R(\counterRMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterR_reg[6] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(plusOp[6]),
        .Q(counterR[6]),
        .R(\counterRMax[7]_i_1_n_0 ));
  FDRE \u_clocks/counterR_reg[7] 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(plusOp[7]),
        .Q(counterR[7]),
        .R(\counterRMax[7]_i_1_n_0 ));
  (* box_type = "PRIMITIVE" *) 
  PLLE2_ADV #(
    .BANDWIDTH("OPTIMIZED"),
    .CLKFBOUT_MULT(12),
    .CLKFBOUT_PHASE(0.000000),
    .CLKIN1_PERIOD(12.500000),
    .CLKIN2_PERIOD(12.500000),
    .CLKOUT0_DIVIDE(12),
    .CLKOUT0_DUTY_CYCLE(0.500000),
    .CLKOUT0_PHASE(0.000000),
    .CLKOUT1_DIVIDE(12),
    .CLKOUT1_DUTY_CYCLE(0.500000),
    .CLKOUT1_PHASE(0.000000),
    .CLKOUT2_DIVIDE(24),
    .CLKOUT2_DUTY_CYCLE(0.500000),
    .CLKOUT2_PHASE(0.000000),
    .CLKOUT3_DIVIDE(1),
    .CLKOUT3_DUTY_CYCLE(0.500000),
    .CLKOUT3_PHASE(0.000000),
    .CLKOUT4_DIVIDE(1),
    .CLKOUT4_DUTY_CYCLE(0.500000),
    .CLKOUT4_PHASE(0.000000),
    .CLKOUT5_DIVIDE(1),
    .CLKOUT5_DUTY_CYCLE(0.500000),
    .CLKOUT5_PHASE(0.000000),
    .COMPENSATION("INTERNAL"),
    .DIVCLK_DIVIDE(1),
    .IS_CLKINSEL_INVERTED(1'b0),
    .IS_PWRDWN_INVERTED(1'b0),
    .IS_RST_INVERTED(1'b0),
    .REF_JITTER1(0.000000),
    .REF_JITTER2(0.000000),
    .STARTUP_WAIT("FALSE")) 
    \u_clocks/mmcm_mclk 
       (.CLKFBIN(\u_clocks/CLKFBIN ),
        .CLKFBOUT(\u_clocks/CLKFBIN ),
        .CLKIN1(LMS_DIQ2_MCLK),
        .CLKIN2(LMS_DIQ1_MCLK),
        .CLKINSEL(1'b1),
        .CLKOUT0(\u_clocks/mmcm_clk_2x ),
        .CLKOUT1(\u_clocks/mmcm_clk_fb ),
        .CLKOUT2(\u_clocks/mmcm_clk_1x ),
        .CLKOUT3(\NLW_u_clocks/mmcm_mclk_CLKOUT3_UNCONNECTED ),
        .CLKOUT4(\NLW_u_clocks/mmcm_mclk_CLKOUT4_UNCONNECTED ),
        .CLKOUT5(\NLW_u_clocks/mmcm_mclk_CLKOUT5_UNCONNECTED ),
        .DADDR(SYS_config_data[22:16]),
        .DCLK(SYS_clk),
        .DEN(SYS_config_data[30]),
        .DI(SYS_config_data[15:0]),
        .DO({\u_clocks/mmcm_mclk_n_9 ,\u_clocks/mmcm_mclk_n_10 ,\u_clocks/mmcm_mclk_n_11 ,\u_clocks/mmcm_mclk_n_12 ,\u_clocks/mmcm_mclk_n_13 ,\u_clocks/mmcm_mclk_n_14 ,\u_clocks/mmcm_mclk_n_15 ,\u_clocks/mmcm_mclk_n_16 ,\u_clocks/mmcm_mclk_n_17 ,\u_clocks/mmcm_mclk_n_18 ,\u_clocks/mmcm_mclk_n_19 ,\u_clocks/mmcm_mclk_n_20 ,\u_clocks/mmcm_mclk_n_21 ,\u_clocks/mmcm_mclk_n_22 ,\u_clocks/mmcm_mclk_n_23 ,\u_clocks/mmcm_mclk_n_24 }),
        .DRDY(\u_clocks/mmcm_mclk_n_7 ),
        .DWE(SYS_config_write),
        .LOCKED(\NLW_u_clocks/mmcm_mclk_LOCKED_UNCONNECTED ),
        .PWRDWN(1'b0),
        .RST(EXT_rst));
  (* OPT_MODIFIED = "MLO " *) 
  (* __SRVAL = "TRUE" *) 
  (* box_type = "PRIMITIVE" *) 
  ODDR #(
    .DDR_CLK_EDGE("OPPOSITE_EDGE"),
    .INIT(1'b0),
    .IS_C_INVERTED(1'b0),
    .IS_D1_INVERTED(1'b0),
    .IS_D2_INVERTED(1'b0),
    .SRTYPE("SYNC")) 
    \u_clocks/oddr_fclk 
       (.C(\u_clocks/data_clk_fb ),
        .CE(1'b1),
        .D1(1'b1),
        .D2(1'b0),
        .Q(LMS_DIQ1_FCLK),
        .R(\NLW_u_clocks/oddr_fclk_R_UNCONNECTED ),
        .S(\NLW_u_clocks/oddr_fclk_S_UNCONNECTED ));
  FDRE #(
    .INIT(1'b0)) 
    \u_clocks/samp_clk_1x_reg 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(samp_clk_1x_i_1_n_0),
        .Q(\u_clocks/samp_clk_1x ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_clocks/samp_clk_2x_reg 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(samp_clk_2x_i_1_n_0),
        .Q(\u_clocks/samp_clk_2x ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_clocks/samp_clk_fb_reg 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(samp_clk_fb_i_1_n_0),
        .Q(\u_clocks/samp_clk_fb_reg_n_0 ),
        .R(1'b0));
  FDRE \u_tester/count_reg[0] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(p_0_in__0[0]),
        .Q(\u_tester/count_reg__0 [0]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/count_reg[1] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(p_0_in__0[1]),
        .Q(\u_tester/count_reg__0 [1]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/count_reg[2] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(p_0_in__0[2]),
        .Q(\u_tester/count_reg__0 [2]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/count_reg[3] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(p_0_in__0[3]),
        .Q(\u_tester/count_reg__0 [3]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/count_reg[4] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(p_0_in__0[4]),
        .Q(\u_tester/count_reg__0 [4]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/count_reg[5] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(p_0_in__0[5]),
        .Q(\u_tester/count_reg__0 [5]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/count_reg[6] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(p_0_in__0[6]),
        .Q(\u_tester/count_reg__0 [6]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/err_reg[10] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[10]_i_1_n_0 ),
        .Q(prbs_e[10]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[11] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[11]_i_1_n_0 ),
        .Q(prbs_e[11]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[12] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[12]_i_1_n_0 ),
        .Q(prbs_e[12]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[13] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[13]_i_1_n_0 ),
        .Q(prbs_e[13]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[14] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[14]_i_1_n_0 ),
        .Q(prbs_e[14]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[15] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[15]_i_1_n_0 ),
        .Q(prbs_e[15]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[20] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[20]_i_1_n_0 ),
        .Q(prbs_e[20]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[21] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[21]_i_1_n_0 ),
        .Q(prbs_e[21]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[22] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[22]_i_1_n_0 ),
        .Q(prbs_e[22]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[23] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[23]_i_1_n_0 ),
        .Q(prbs_e[23]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[24] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[24]_i_1_n_0 ),
        .Q(prbs_e[24]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[25] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[25]_i_1_n_0 ),
        .Q(prbs_e[25]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[26] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[26]_i_1_n_0 ),
        .Q(prbs_e[26]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[27] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[27]_i_1_n_0 ),
        .Q(prbs_e[27]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[28] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[28]_i_1_n_0 ),
        .Q(prbs_e[28]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[29] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[29]_i_1_n_0 ),
        .Q(prbs_e[29]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[30] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[30]_i_1_n_0 ),
        .Q(prbs_e[30]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[31] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[31]_i_1_n_0 ),
        .Q(prbs_e[31]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[36] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[36]_i_1_n_0 ),
        .Q(prbs_e[36]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[37] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[37]_i_1_n_0 ),
        .Q(prbs_e[37]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[38] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[38]_i_1_n_0 ),
        .Q(prbs_e[38]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[39] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[39]_i_1_n_0 ),
        .Q(prbs_e[39]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[40] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[40]_i_1_n_0 ),
        .Q(prbs_e[40]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[41] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[41]_i_1_n_0 ),
        .Q(prbs_e[41]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[42] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[42]_i_1_n_0 ),
        .Q(prbs_e[42]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[43] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[43]_i_1_n_0 ),
        .Q(prbs_e[43]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[44] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[44]_i_1_n_0 ),
        .Q(prbs_e[44]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[45] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[45]_i_1_n_0 ),
        .Q(prbs_e[45]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[46] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[46]_i_1_n_0 ),
        .Q(prbs_e[46]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[47] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[47]_i_1_n_0 ),
        .Q(prbs_e[47]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[4] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[4]_i_1_n_0 ),
        .Q(prbs_e[4]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[52] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[52]_i_1_n_0 ),
        .Q(prbs_e[52]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[53] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[53]_i_1_n_0 ),
        .Q(prbs_e[53]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[54] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[54]_i_1_n_0 ),
        .Q(prbs_e[54]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[55] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[55]_i_1_n_0 ),
        .Q(prbs_e[55]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[56] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[56]_i_1_n_0 ),
        .Q(prbs_e[56]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[57] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[57]_i_1_n_0 ),
        .Q(prbs_e[57]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[58] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[58]_i_1_n_0 ),
        .Q(prbs_e[58]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[59] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[59]_i_1_n_0 ),
        .Q(prbs_e[59]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[5] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[5]_i_1_n_0 ),
        .Q(prbs_e[5]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[60] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[60]_i_1_n_0 ),
        .Q(prbs_e[60]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[61] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[61]_i_1_n_0 ),
        .Q(prbs_e[61]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[62] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[62]_i_1_n_0 ),
        .Q(prbs_e[62]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[63] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[63]_i_2_n_0 ),
        .Q(prbs_e[63]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[6] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[6]_i_1_n_0 ),
        .Q(prbs_e[6]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[7] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[7]_i_1_n_0 ),
        .Q(prbs_e[7]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[8] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[8]_i_1_n_0 ),
        .Q(prbs_e[8]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/err_reg[9] 
       (.C(DATA_clk),
        .CE(prbs_stat[27]),
        .D(\err[9]_i_1_n_0 ),
        .Q(prbs_e[9]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/mask_count_reg[10] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[9] ),
        .Q(\u_tester/mask_count_reg_n_0_[10] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[11] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[10] ),
        .Q(\u_tester/mask_count_reg_n_0_[11] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[12] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[11] ),
        .Q(\u_tester/mask_count_reg_n_0_[12] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[13] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[12] ),
        .Q(\u_tester/mask_count_reg_n_0_[13] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[14] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[13] ),
        .Q(\u_tester/mask_count_reg_n_0_[14] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[15] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[14] ),
        .Q(\u_tester/mask_count_reg_n_0_[15] ),
        .R(\u_tester/mask_count ));
  (* srl_bus_name = "\u_tester/mask_count_reg " *) 
  (* srl_name = "\u_tester/mask_count_reg[18]_srl3___u_tester_mask_count_reg_r_4 " *) 
  SRL16E \u_tester/mask_count_reg[18]_srl3___u_tester_mask_count_reg_r_4 
       (.A0(1'b0),
        .A1(1'b1),
        .A2(1'b0),
        .A3(1'b0),
        .CE(prbs_ctrl[12]),
        .CLK(DATA_clk),
        .D(\u_tester/mask_count_reg_n_0_[15] ),
        .Q(\u_tester/mask_count_reg[18]_srl3___u_tester_mask_count_reg_r_4_n_0 ));
  FDRE \u_tester/mask_count_reg[19]_u_tester_mask_count_reg_r_5 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg[18]_srl3___u_tester_mask_count_reg_r_4_n_0 ),
        .Q(\u_tester/mask_count_reg[19]_u_tester_mask_count_reg_r_5_n_0 ),
        .R(1'b0));
  FDRE \u_tester/mask_count_reg[20] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_gate__1_n_0 ),
        .Q(\u_tester/mask_count_reg_n_0_[20] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[21] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[20] ),
        .Q(\u_tester/mask_count_reg_n_0_[21] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[22] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[21] ),
        .Q(\u_tester/mask_count_reg_n_0_[22] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[23] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[22] ),
        .Q(\u_tester/mask_count_reg_n_0_[23] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[24] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[23] ),
        .Q(\u_tester/mask_count_reg_n_0_[24] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[25] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[24] ),
        .Q(\u_tester/mask_count_reg_n_0_[25] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[26] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[25] ),
        .Q(\u_tester/mask_count_reg_n_0_[26] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[27] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[26] ),
        .Q(\u_tester/mask_count_reg_n_0_[27] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[28] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[27] ),
        .Q(\u_tester/mask_count_reg_n_0_[28] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[29] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[28] ),
        .Q(\u_tester/mask_count_reg_n_0_[29] ),
        .R(\u_tester/mask_count ));
  (* srl_bus_name = "\u_tester/mask_count_reg " *) 
  (* srl_name = "\u_tester/mask_count_reg[2]_srl3___u_tester_mask_count_reg_r_4 " *) 
  SRL16E \u_tester/mask_count_reg[2]_srl3___u_tester_mask_count_reg_r_4 
       (.A0(1'b0),
        .A1(1'b1),
        .A2(1'b0),
        .A3(1'b0),
        .CE(prbs_ctrl[12]),
        .CLK(DATA_clk),
        .D(prbs_ctrl[12]),
        .Q(\u_tester/mask_count_reg[2]_srl3___u_tester_mask_count_reg_r_4_n_0 ));
  FDRE \u_tester/mask_count_reg[30] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[29] ),
        .Q(\u_tester/mask_count_reg_n_0_[30] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[31] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[30] ),
        .Q(\u_tester/mask_count_reg_n_0_[31] ),
        .R(\u_tester/mask_count ));
  (* srl_bus_name = "\u_tester/mask_count_reg " *) 
  (* srl_name = "\u_tester/mask_count_reg[34]_srl3___u_tester_mask_count_reg_r_4 " *) 
  SRL16E \u_tester/mask_count_reg[34]_srl3___u_tester_mask_count_reg_r_4 
       (.A0(1'b0),
        .A1(1'b1),
        .A2(1'b0),
        .A3(1'b0),
        .CE(prbs_ctrl[12]),
        .CLK(DATA_clk),
        .D(\u_tester/mask_count_reg_n_0_[31] ),
        .Q(\u_tester/mask_count_reg[34]_srl3___u_tester_mask_count_reg_r_4_n_0 ));
  FDRE \u_tester/mask_count_reg[35]_u_tester_mask_count_reg_r_5 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg[34]_srl3___u_tester_mask_count_reg_r_4_n_0 ),
        .Q(\u_tester/mask_count_reg[35]_u_tester_mask_count_reg_r_5_n_0 ),
        .R(1'b0));
  FDRE \u_tester/mask_count_reg[36] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_gate__0_n_0 ),
        .Q(\u_tester/mask_count_reg_n_0_[36] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[37] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[36] ),
        .Q(\u_tester/mask_count_reg_n_0_[37] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[38] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[37] ),
        .Q(\u_tester/mask_count_reg_n_0_[38] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[39] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[38] ),
        .Q(\u_tester/mask_count_reg_n_0_[39] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[3]_u_tester_mask_count_reg_r_5 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg[2]_srl3___u_tester_mask_count_reg_r_4_n_0 ),
        .Q(\u_tester/mask_count_reg[3]_u_tester_mask_count_reg_r_5_n_0 ),
        .R(1'b0));
  FDRE \u_tester/mask_count_reg[40] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[39] ),
        .Q(\u_tester/mask_count_reg_n_0_[40] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[41] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[40] ),
        .Q(\u_tester/mask_count_reg_n_0_[41] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[42] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[41] ),
        .Q(\u_tester/mask_count_reg_n_0_[42] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[43] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[42] ),
        .Q(\u_tester/mask_count_reg_n_0_[43] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[44] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[43] ),
        .Q(\u_tester/mask_count_reg_n_0_[44] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[45] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[44] ),
        .Q(\u_tester/mask_count_reg_n_0_[45] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[46] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[45] ),
        .Q(\u_tester/mask_count_reg_n_0_[46] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[47] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[46] ),
        .Q(\u_tester/mask_count_reg_n_0_[47] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[4] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_gate__2_n_0 ),
        .Q(\u_tester/mask_count_reg_n_0_[4] ),
        .R(\u_tester/mask_count ));
  (* srl_bus_name = "\u_tester/mask_count_reg " *) 
  (* srl_name = "\u_tester/mask_count_reg[50]_srl3___u_tester_mask_count_reg_r_4 " *) 
  SRL16E \u_tester/mask_count_reg[50]_srl3___u_tester_mask_count_reg_r_4 
       (.A0(1'b0),
        .A1(1'b1),
        .A2(1'b0),
        .A3(1'b0),
        .CE(prbs_ctrl[12]),
        .CLK(DATA_clk),
        .D(\u_tester/mask_count_reg_n_0_[47] ),
        .Q(\u_tester/mask_count_reg[50]_srl3___u_tester_mask_count_reg_r_4_n_0 ));
  FDRE \u_tester/mask_count_reg[51]_u_tester_mask_count_reg_r_5 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg[50]_srl3___u_tester_mask_count_reg_r_4_n_0 ),
        .Q(\u_tester/mask_count_reg[51]_u_tester_mask_count_reg_r_5_n_0 ),
        .R(1'b0));
  FDRE \u_tester/mask_count_reg[52] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_gate_n_0 ),
        .Q(\u_tester/mask_count_reg_n_0_[52] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[53] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[52] ),
        .Q(\u_tester/mask_count_reg_n_0_[53] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[54] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[53] ),
        .Q(\u_tester/mask_count_reg_n_0_[54] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[55] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[54] ),
        .Q(\u_tester/mask_count_reg_n_0_[55] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[56] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[55] ),
        .Q(\u_tester/mask_count_reg_n_0_[56] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[57] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[56] ),
        .Q(\u_tester/mask_count_reg_n_0_[57] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[58] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[57] ),
        .Q(\u_tester/mask_count_reg_n_0_[58] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[59] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[58] ),
        .Q(\u_tester/mask_count_reg_n_0_[59] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[5] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[4] ),
        .Q(\u_tester/mask_count_reg_n_0_[5] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[60] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[59] ),
        .Q(\u_tester/mask_count_reg_n_0_[60] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[61] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[60] ),
        .Q(\u_tester/mask_count_reg_n_0_[61] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[62] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[61] ),
        .Q(\u_tester/mask_count_reg_n_0_[62] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[63] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[62] ),
        .Q(\u_tester/mask_count_reg_n_0_[63] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[6] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[5] ),
        .Q(\u_tester/mask_count_reg_n_0_[6] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[7] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[6] ),
        .Q(\u_tester/mask_count_reg_n_0_[7] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[8] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[7] ),
        .Q(\u_tester/mask_count_reg_n_0_[8] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[9] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_n_0_[8] ),
        .Q(\u_tester/mask_count_reg_n_0_[9] ),
        .R(\u_tester/mask_count ));
  (* SOFT_HLUTNM = "soft_lutpair60" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \u_tester/mask_count_reg_gate 
       (.I0(\u_tester/mask_count_reg[51]_u_tester_mask_count_reg_r_5_n_0 ),
        .I1(\u_tester/mask_count_reg_r_5_n_0 ),
        .O(\u_tester/mask_count_reg_gate_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair61" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \u_tester/mask_count_reg_gate__0 
       (.I0(\u_tester/mask_count_reg[35]_u_tester_mask_count_reg_r_5_n_0 ),
        .I1(\u_tester/mask_count_reg_r_5_n_0 ),
        .O(\u_tester/mask_count_reg_gate__0_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair61" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \u_tester/mask_count_reg_gate__1 
       (.I0(\u_tester/mask_count_reg[19]_u_tester_mask_count_reg_r_5_n_0 ),
        .I1(\u_tester/mask_count_reg_r_5_n_0 ),
        .O(\u_tester/mask_count_reg_gate__1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair60" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \u_tester/mask_count_reg_gate__2 
       (.I0(\u_tester/mask_count_reg[3]_u_tester_mask_count_reg_r_5_n_0 ),
        .I1(\u_tester/mask_count_reg_r_5_n_0 ),
        .O(\u_tester/mask_count_reg_gate__2_n_0 ));
  FDRE \u_tester/mask_count_reg_r 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(1'b1),
        .Q(\u_tester/mask_count_reg_r_n_0 ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg_r_3 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_r_n_0 ),
        .Q(\u_tester/mask_count_reg_r_3_n_0 ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg_r_4 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_r_3_n_0 ),
        .Q(\u_tester/mask_count_reg_r_4_n_0 ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg_r_5 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/mask_count_reg_r_4_n_0 ),
        .Q(\u_tester/mask_count_reg_r_5_n_0 ),
        .R(\u_tester/mask_count ));
  CARRY4 \u_tester/rx_err0_carry 
       (.CI(1'b0),
        .CO({\u_tester/rx_err0_carry_n_0 ,\u_tester/rx_err0_carry_n_1 ,\u_tester/rx_err0_carry_n_2 ,\u_tester/rx_err0_carry_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b1,1'b1,1'b1,1'b1}),
        .O(\NLW_u_tester/rx_err0_carry_O_UNCONNECTED [3:0]),
        .S({rx_err0_carry_i_1_n_0,rx_err0_carry_i_2_n_0,rx_err0_carry_i_3_n_0,rx_err0_carry_i_4_n_0}));
  CARRY4 \u_tester/rx_err0_carry__0 
       (.CI(\u_tester/rx_err0_carry_n_0 ),
        .CO({\u_tester/rx_err0_carry__0_n_0 ,\u_tester/rx_err0_carry__0_n_1 ,\u_tester/rx_err0_carry__0_n_2 ,\u_tester/rx_err0_carry__0_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b1,1'b1,1'b1,1'b1}),
        .O(\NLW_u_tester/rx_err0_carry__0_O_UNCONNECTED [3:0]),
        .S({rx_err0_carry__0_i_1_n_0,rx_err0_carry__0_i_2_n_0,rx_err0_carry__0_i_3_n_0,rx_err0_carry__0_i_4_n_0}));
  CARRY4 \u_tester/rx_err0_carry__1 
       (.CI(\u_tester/rx_err0_carry__0_n_0 ),
        .CO({\u_tester/rx_err0_carry__1_n_0 ,\u_tester/rx_err0_carry__1_n_1 ,\u_tester/rx_err0_carry__1_n_2 ,\u_tester/rx_err0_carry__1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b1,1'b1,1'b1,1'b1}),
        .O(\NLW_u_tester/rx_err0_carry__1_O_UNCONNECTED [3:0]),
        .S({rx_err0_carry__1_i_1_n_0,rx_err0_carry__1_i_2_n_0,rx_err0_carry__1_i_3_n_0,rx_err0_carry__1_i_4_n_0}));
  CARRY4 \u_tester/rx_err0_carry__2 
       (.CI(\u_tester/rx_err0_carry__1_n_0 ),
        .CO({\u_tester/rx_err0_carry__2_n_0 ,\u_tester/rx_err0_carry__2_n_1 ,\u_tester/rx_err0_carry__2_n_2 ,\u_tester/rx_err0_carry__2_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b1,1'b1,1'b1,1'b1}),
        .O(\NLW_u_tester/rx_err0_carry__2_O_UNCONNECTED [3:0]),
        .S({rx_err0_carry__2_i_1_n_0,rx_err0_carry__2_i_2_n_0,rx_err0_carry__2_i_3_n_0,rx_err0_carry__2_i_4_n_0}));
  CARRY4 \u_tester/rx_err0_carry__3 
       (.CI(\u_tester/rx_err0_carry__2_n_0 ),
        .CO({\NLW_u_tester/rx_err0_carry__3_CO_UNCONNECTED [3],\u_tester/rx_err0 ,\u_tester/rx_err0_carry__3_n_2 ,\u_tester/rx_err0_carry__3_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b1,1'b1,1'b1}),
        .O(\NLW_u_tester/rx_err0_carry__3_O_UNCONNECTED [3:0]),
        .S({1'b0,rx_err0_carry__3_i_1_n_0,rx_err0_carry__3_i_2_n_0,rx_err0_carry__3_i_3_n_0}));
  FDRE \u_tester/rx_err_cnt_reg[0] 
       (.C(DATA_clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__1[0]),
        .Q(prbs_stat[0]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[1] 
       (.C(DATA_clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__1[1]),
        .Q(prbs_stat[1]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[2] 
       (.C(DATA_clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__1[2]),
        .Q(prbs_stat[2]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[3] 
       (.C(DATA_clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__1[3]),
        .Q(prbs_stat[3]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[4] 
       (.C(DATA_clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__1[4]),
        .Q(prbs_stat[4]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[5] 
       (.C(DATA_clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__1[5]),
        .Q(prbs_stat[5]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[6] 
       (.C(DATA_clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__1[6]),
        .Q(prbs_stat[6]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[7] 
       (.C(DATA_clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__1[7]),
        .Q(prbs_stat[7]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/rx_err_s_reg 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(rx_err_s_i_1_n_0),
        .Q(prbs_stat[21]),
        .R(1'b0));
  FDRE \u_tester/slip_cnt_reg[0] 
       (.C(DATA_clk),
        .CE(\u_tester/rx_slip ),
        .D(p_0_in__2[0]),
        .Q(prbs_stat[16]),
        .R(\slip_cnt[3]_i_1_n_0 ));
  FDRE \u_tester/slip_cnt_reg[1] 
       (.C(DATA_clk),
        .CE(\u_tester/rx_slip ),
        .D(p_0_in__2[1]),
        .Q(prbs_stat[17]),
        .R(\slip_cnt[3]_i_1_n_0 ));
  FDRE \u_tester/slip_cnt_reg[2] 
       (.C(DATA_clk),
        .CE(\u_tester/rx_slip ),
        .D(p_0_in__2[2]),
        .Q(prbs_stat[18]),
        .R(\slip_cnt[3]_i_1_n_0 ));
  FDRE \u_tester/slip_cnt_reg[3] 
       (.C(DATA_clk),
        .CE(\u_tester/rx_slip ),
        .D(p_0_in__2[3]),
        .Q(prbs_stat[19]),
        .R(\slip_cnt[3]_i_1_n_0 ));
  FDRE \u_tester/st_cur_reg[0] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\st_cur[0]_i_1_n_0 ),
        .Q(\u_tester/st_cur [0]),
        .R(1'b0));
  FDRE \u_tester/st_cur_reg[1] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\st_cur[1]_i_1_n_0 ),
        .Q(\u_tester/st_cur [1]),
        .R(1'b0));
  FDRE \u_tester/tx_cnt_reg[0] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[0]_i_2_n_7 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[0] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[10] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[8]_i_1_n_5 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[10] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[11] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[8]_i_1_n_4 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[11] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[12] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[12]_i_1_n_7 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[12] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[13] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[12]_i_1_n_6 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[13] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[14] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[12]_i_1_n_5 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[14] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[15] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[12]_i_1_n_4 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[15] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[16] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[16]_i_1_n_7 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[16] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[17] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[16]_i_1_n_6 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[17] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[18] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[16]_i_1_n_5 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[18] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[19] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[16]_i_1_n_4 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[19] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[1] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[0]_i_2_n_6 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[1] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[20] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[20]_i_1_n_7 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[20] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[21] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[20]_i_1_n_6 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[21] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[22] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[20]_i_1_n_5 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[22] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[23] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[20]_i_1_n_4 ),
        .Q(prbs_stat[8]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[24] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[24]_i_1_n_7 ),
        .Q(prbs_stat[9]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[25] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[24]_i_1_n_6 ),
        .Q(prbs_stat[10]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[26] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[24]_i_1_n_5 ),
        .Q(prbs_stat[11]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[27] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[24]_i_1_n_4 ),
        .Q(prbs_stat[12]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[28] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[28]_i_1_n_7 ),
        .Q(prbs_stat[13]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[29] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[28]_i_1_n_6 ),
        .Q(prbs_stat[14]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[2] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[0]_i_2_n_5 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[2] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[30] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[28]_i_1_n_5 ),
        .Q(prbs_stat[15]),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[3] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[0]_i_2_n_4 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[3] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[4] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[4]_i_1_n_7 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[4] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[5] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[4]_i_1_n_6 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[5] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[6] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[4]_i_1_n_5 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[6] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[7] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[4]_i_1_n_4 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[7] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[8] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[8]_i_1_n_7 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[8] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[9] 
       (.C(DATA_clk),
        .CE(\u_tester/tx_cnt0 ),
        .D(\tx_cnt_reg[8]_i_1_n_6 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[9] ),
        .R(\err[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[10] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [9]),
        .Q(\u_tester/chk_out [10]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[11] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [10]),
        .Q(\u_tester/chk_out [11]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[12] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [11]),
        .Q(\u_tester/chk_out [12]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[13] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [12]),
        .Q(\u_tester/chk_out [13]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[14] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [13]),
        .Q(\u_tester/chk_out [14]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[15] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [14]),
        .Q(\u_tester/chk_out [15]),
        .R(\out[63]_i_1_n_0 ));
  (* srl_bus_name = "\u_tester/u_chk/out_reg " *) 
  (* srl_name = "\u_tester/u_chk/out_reg[18]_srl3___u_tester_u_chk_out_reg_r_1 " *) 
  SRL16E \u_tester/u_chk/out_reg[18]_srl3___u_tester_u_chk_out_reg_r_1 
       (.A0(1'b0),
        .A1(1'b1),
        .A2(1'b0),
        .A3(1'b0),
        .CE(\u_tester/chk_en ),
        .CLK(DATA_clk),
        .D(\u_tester/chk_out [15]),
        .Q(\u_tester/u_chk/out_reg[18]_srl3___u_tester_u_chk_out_reg_r_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[19]_u_tester_u_chk_out_reg_r_2 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/u_chk/out_reg[18]_srl3___u_tester_u_chk_out_reg_r_1_n_0 ),
        .Q(\u_tester/u_chk/out_reg[19]_u_tester_u_chk_out_reg_r_2_n_0 ),
        .R(1'b0));
  FDRE \u_tester/u_chk/out_reg[20] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/u_chk/out_reg_gate__1_n_0 ),
        .Q(\u_tester/chk_out [20]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[21] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [20]),
        .Q(\u_tester/chk_out [21]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[22] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [21]),
        .Q(\u_tester/chk_out [22]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[23] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [22]),
        .Q(\u_tester/chk_out [23]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[24] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [23]),
        .Q(\u_tester/chk_out [24]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[25] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [24]),
        .Q(\u_tester/chk_out [25]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[26] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [25]),
        .Q(\u_tester/chk_out [26]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[27] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [26]),
        .Q(\u_tester/chk_out [27]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[28] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [27]),
        .Q(\u_tester/chk_out [28]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[29] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [28]),
        .Q(\u_tester/chk_out [29]),
        .R(\out[63]_i_1_n_0 ));
  (* srl_bus_name = "\u_tester/u_chk/out_reg " *) 
  (* srl_name = "\u_tester/u_chk/out_reg[2]_srl3___u_tester_u_chk_out_reg_r_1 " *) 
  SRL16E \u_tester/u_chk/out_reg[2]_srl3___u_tester_u_chk_out_reg_r_1 
       (.A0(1'b0),
        .A1(1'b1),
        .A2(1'b0),
        .A3(1'b0),
        .CE(\u_tester/chk_en ),
        .CLK(DATA_clk),
        .D(\out_reg[2]_srl3___u_tester_u_chk_out_reg_r_1_i_1_n_0 ),
        .Q(\u_tester/u_chk/out_reg[2]_srl3___u_tester_u_chk_out_reg_r_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[30] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [29]),
        .Q(\u_tester/chk_out [30]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[31] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [30]),
        .Q(\u_tester/chk_out [31]),
        .R(\out[63]_i_1_n_0 ));
  (* srl_bus_name = "\u_tester/u_chk/out_reg " *) 
  (* srl_name = "\u_tester/u_chk/out_reg[34]_srl3___u_tester_u_chk_out_reg_r_1 " *) 
  SRL16E \u_tester/u_chk/out_reg[34]_srl3___u_tester_u_chk_out_reg_r_1 
       (.A0(1'b0),
        .A1(1'b1),
        .A2(1'b0),
        .A3(1'b0),
        .CE(\u_tester/chk_en ),
        .CLK(DATA_clk),
        .D(\u_tester/chk_out [31]),
        .Q(\u_tester/u_chk/out_reg[34]_srl3___u_tester_u_chk_out_reg_r_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[35]_u_tester_u_chk_out_reg_r_2 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/u_chk/out_reg[34]_srl3___u_tester_u_chk_out_reg_r_1_n_0 ),
        .Q(\u_tester/u_chk/out_reg[35]_u_tester_u_chk_out_reg_r_2_n_0 ),
        .R(1'b0));
  FDRE \u_tester/u_chk/out_reg[36] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/u_chk/out_reg_gate__0_n_0 ),
        .Q(\u_tester/chk_out [36]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[37] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [36]),
        .Q(\u_tester/chk_out [37]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[38] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [37]),
        .Q(\u_tester/chk_out [38]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[39] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [38]),
        .Q(\u_tester/chk_out [39]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[3]_u_tester_u_chk_out_reg_r_2 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/u_chk/out_reg[2]_srl3___u_tester_u_chk_out_reg_r_1_n_0 ),
        .Q(\u_tester/u_chk/out_reg[3]_u_tester_u_chk_out_reg_r_2_n_0 ),
        .R(1'b0));
  FDRE \u_tester/u_chk/out_reg[40] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [39]),
        .Q(\u_tester/chk_out [40]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[41] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [40]),
        .Q(\u_tester/chk_out [41]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[42] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [41]),
        .Q(\u_tester/chk_out [42]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[43] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [42]),
        .Q(\u_tester/chk_out [43]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[44] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [43]),
        .Q(\u_tester/chk_out [44]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[45] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [44]),
        .Q(\u_tester/chk_out [45]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[46] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [45]),
        .Q(\u_tester/chk_out [46]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[47] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [46]),
        .Q(\u_tester/chk_out [47]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[4] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/u_chk/out_reg_gate__2_n_0 ),
        .Q(\u_tester/chk_out [4]),
        .R(\out[63]_i_1_n_0 ));
  (* srl_bus_name = "\u_tester/u_chk/out_reg " *) 
  (* srl_name = "\u_tester/u_chk/out_reg[50]_srl3___u_tester_u_chk_out_reg_r_1 " *) 
  SRL16E \u_tester/u_chk/out_reg[50]_srl3___u_tester_u_chk_out_reg_r_1 
       (.A0(1'b0),
        .A1(1'b1),
        .A2(1'b0),
        .A3(1'b0),
        .CE(\u_tester/chk_en ),
        .CLK(DATA_clk),
        .D(\u_tester/chk_out [47]),
        .Q(\u_tester/u_chk/out_reg[50]_srl3___u_tester_u_chk_out_reg_r_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[51]_u_tester_u_chk_out_reg_r_2 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/u_chk/out_reg[50]_srl3___u_tester_u_chk_out_reg_r_1_n_0 ),
        .Q(\u_tester/u_chk/out_reg[51]_u_tester_u_chk_out_reg_r_2_n_0 ),
        .R(1'b0));
  FDRE \u_tester/u_chk/out_reg[52] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/u_chk/out_reg_gate_n_0 ),
        .Q(\u_tester/chk_out [52]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[53] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [52]),
        .Q(\u_tester/chk_out [53]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[54] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [53]),
        .Q(\u_tester/chk_out [54]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[55] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [54]),
        .Q(\u_tester/chk_out [55]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[56] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [55]),
        .Q(\u_tester/chk_out [56]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[57] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [56]),
        .Q(\u_tester/chk_out [57]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[58] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [57]),
        .Q(\u_tester/chk_out [58]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[59] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [58]),
        .Q(\u_tester/chk_out [59]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[5] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [4]),
        .Q(\u_tester/chk_out [5]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[60] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [59]),
        .Q(\u_tester/chk_out [60]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[61] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [60]),
        .Q(\u_tester/chk_out [61]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[62] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [61]),
        .Q(\u_tester/chk_out [62]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[63] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [62]),
        .Q(\u_tester/chk_out [63]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[6] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [5]),
        .Q(\u_tester/chk_out [6]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[7] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [6]),
        .Q(\u_tester/chk_out [7]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[8] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [7]),
        .Q(\u_tester/chk_out [8]),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[9] 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [8]),
        .Q(\u_tester/chk_out [9]),
        .R(\out[63]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair59" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \u_tester/u_chk/out_reg_gate 
       (.I0(\u_tester/u_chk/out_reg[51]_u_tester_u_chk_out_reg_r_2_n_0 ),
        .I1(\u_tester/u_chk/out_reg_r_2_n_0 ),
        .O(\u_tester/u_chk/out_reg_gate_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair59" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \u_tester/u_chk/out_reg_gate__0 
       (.I0(\u_tester/u_chk/out_reg[35]_u_tester_u_chk_out_reg_r_2_n_0 ),
        .I1(\u_tester/u_chk/out_reg_r_2_n_0 ),
        .O(\u_tester/u_chk/out_reg_gate__0_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair62" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \u_tester/u_chk/out_reg_gate__1 
       (.I0(\u_tester/u_chk/out_reg[19]_u_tester_u_chk_out_reg_r_2_n_0 ),
        .I1(\u_tester/u_chk/out_reg_r_2_n_0 ),
        .O(\u_tester/u_chk/out_reg_gate__1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair62" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \u_tester/u_chk/out_reg_gate__2 
       (.I0(\u_tester/u_chk/out_reg[3]_u_tester_u_chk_out_reg_r_2_n_0 ),
        .I1(\u_tester/u_chk/out_reg_r_2_n_0 ),
        .O(\u_tester/u_chk/out_reg_gate__2_n_0 ));
  FDRE \u_tester/u_chk/out_reg_r 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(1'b1),
        .Q(\u_tester/u_chk/out_reg_r_n_0 ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg_r_0 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/u_chk/out_reg_r_n_0 ),
        .Q(\u_tester/u_chk/out_reg_r_0_n_0 ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg_r_1 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/u_chk/out_reg_r_0_n_0 ),
        .Q(\u_tester/u_chk/out_reg_r_1_n_0 ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg_r_2 
       (.C(DATA_clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/u_chk/out_reg_r_1_n_0 ),
        .Q(\u_tester/u_chk/out_reg_r_2_n_0 ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[0] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(p_0_out),
        .Q(\u_tester/u_gen/out_reg_n_0_[0] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[10] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[9] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[10] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[11] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[10] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[11] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[12] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[11] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[12] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[13] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[12] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[13] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[14] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[13] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[14] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[15] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[14] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[15] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[16] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[15] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[16] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[17] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[16] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[17] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[18] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[17] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[18] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[19] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[18] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[19] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[1] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[0] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[1] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[20] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[19] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[20] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[21] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[20] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[21] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[22] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[21] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[22] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[23] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[22] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[23] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[24] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[23] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[24] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[25] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[24] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[25] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[26] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[25] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[26] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[27] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[26] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[27] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[28] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[27] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[28] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[29] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[28] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[29] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[2] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[1] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[2] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[30] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[29] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[30] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[31] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[30] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[31] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[32] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[31] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[32] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[33] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[32] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[33] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[34] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[33] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[34] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[35] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[34] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[35] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[36] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[35] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[36] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[37] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[36] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[37] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[38] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[37] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[38] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[39] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[38] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[39] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[3] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[2] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[3] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[40] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[39] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[40] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[41] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[40] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[41] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[42] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[41] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[42] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[43] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[42] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[43] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[44] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[43] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[44] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[45] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[44] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[45] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[46] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[45] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[46] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[47] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[46] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[47] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[48] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[47] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[48] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[49] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[48] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[49] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[4] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[3] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[4] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[50] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[49] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[50] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[51] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[50] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[51] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[52] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[51] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[52] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[53] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[52] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[53] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[54] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[53] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[54] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[55] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[54] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[55] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[56] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[55] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[56] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[57] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[56] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[57] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[58] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[57] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[58] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[59] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[58] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[59] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[5] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[4] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[5] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[60] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[59] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[60] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[61] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[60] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[61] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[62] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[61] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[62] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[63] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[62] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[63] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[6] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[5] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[6] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[7] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[6] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[7] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[8] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[7] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[8] ),
        .R(\out[63]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[9] 
       (.C(DATA_clk),
        .CE(prbs_ctrl[12]),
        .D(\u_tester/u_gen/out_reg_n_0_[8] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[9] ),
        .R(\out[63]_i_1_n_0 ));
endmodule

(* DEST_SYNC_FF = "4" *) (* INIT_SYNC_FF = "0" *) (* SIM_ASSERT_CHK = "0" *) 
(* SRC_INPUT_REG = "1" *) (* VERSION = "0" *) (* WIDTH = "2" *) 
(* XPM_MODULE = "TRUE" *) (* xpm_cdc = "ARRAY_SINGLE" *) 
module xpm_cdc_array_single
   (src_clk,
    src_in,
    dest_clk,
    dest_out);
  input src_clk;
  input [1:0]src_in;
  input dest_clk;
  output [1:0]dest_out;

  wire [1:0]async_path_bit;
  wire dest_clk;
  wire src_clk;
  wire [1:0]src_in;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [1:0]\syncstages_ff[0] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [1:0]\syncstages_ff[1] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [1:0]\syncstages_ff[2] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [1:0]\syncstages_ff[3] ;

  assign dest_out[1:0] = \syncstages_ff[3] ;
  FDRE \src_ff_reg[0] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[0]),
        .Q(async_path_bit[0]),
        .R(1'b0));
  FDRE \src_ff_reg[1] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[1]),
        .Q(async_path_bit[1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[0]),
        .Q(\syncstages_ff[0] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[1]),
        .Q(\syncstages_ff[0] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [0]),
        .Q(\syncstages_ff[1] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [1]),
        .Q(\syncstages_ff[1] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [0]),
        .Q(\syncstages_ff[2] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [1]),
        .Q(\syncstages_ff[2] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [0]),
        .Q(\syncstages_ff[3] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [1]),
        .Q(\syncstages_ff[3] [1]),
        .R(1'b0));
endmodule

(* DEST_SYNC_FF = "4" *) (* INIT_SYNC_FF = "0" *) (* ORIG_REF_NAME = "xpm_cdc_array_single" *) 
(* SIM_ASSERT_CHK = "0" *) (* SRC_INPUT_REG = "1" *) (* VERSION = "0" *) 
(* WIDTH = "3" *) (* XPM_MODULE = "TRUE" *) (* xpm_cdc = "ARRAY_SINGLE" *) 
module xpm_cdc_array_single__parameterized0
   (src_clk,
    src_in,
    dest_clk,
    dest_out);
  input src_clk;
  input [2:0]src_in;
  input dest_clk;
  output [2:0]dest_out;

  wire [2:0]async_path_bit;
  wire dest_clk;
  wire src_clk;
  wire [2:0]src_in;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [2:0]\syncstages_ff[0] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [2:0]\syncstages_ff[1] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [2:0]\syncstages_ff[2] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [2:0]\syncstages_ff[3] ;

  assign dest_out[2:0] = \syncstages_ff[3] ;
  FDRE \src_ff_reg[0] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[0]),
        .Q(async_path_bit[0]),
        .R(1'b0));
  FDRE \src_ff_reg[1] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[1]),
        .Q(async_path_bit[1]),
        .R(1'b0));
  FDRE \src_ff_reg[2] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[2]),
        .Q(async_path_bit[2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[0]),
        .Q(\syncstages_ff[0] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[1]),
        .Q(\syncstages_ff[0] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[2]),
        .Q(\syncstages_ff[0] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [0]),
        .Q(\syncstages_ff[1] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [1]),
        .Q(\syncstages_ff[1] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [2]),
        .Q(\syncstages_ff[1] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [0]),
        .Q(\syncstages_ff[2] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [1]),
        .Q(\syncstages_ff[2] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [2]),
        .Q(\syncstages_ff[2] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [0]),
        .Q(\syncstages_ff[3] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [1]),
        .Q(\syncstages_ff[3] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [2]),
        .Q(\syncstages_ff[3] [2]),
        .R(1'b0));
endmodule

(* DEST_SYNC_FF = "4" *) (* INIT_SYNC_FF = "0" *) (* ORIG_REF_NAME = "xpm_cdc_array_single" *) 
(* SIM_ASSERT_CHK = "0" *) (* SRC_INPUT_REG = "1" *) (* VERSION = "0" *) 
(* WIDTH = "64" *) (* XPM_MODULE = "TRUE" *) (* xpm_cdc = "ARRAY_SINGLE" *) 
module xpm_cdc_array_single__parameterized1
   (src_clk,
    src_in,
    dest_clk,
    dest_out);
  input src_clk;
  input [63:0]src_in;
  input dest_clk;
  output [63:0]dest_out;

  wire [63:0]async_path_bit;
  wire dest_clk;
  wire src_clk;
  wire [63:0]src_in;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [63:0]\syncstages_ff[0] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [63:0]\syncstages_ff[1] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [63:0]\syncstages_ff[2] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [63:0]\syncstages_ff[3] ;

  assign dest_out[63:0] = \syncstages_ff[3] ;
  FDRE \src_ff_reg[0] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[0]),
        .Q(async_path_bit[0]),
        .R(1'b0));
  FDRE \src_ff_reg[10] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[10]),
        .Q(async_path_bit[10]),
        .R(1'b0));
  FDRE \src_ff_reg[11] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[11]),
        .Q(async_path_bit[11]),
        .R(1'b0));
  FDRE \src_ff_reg[12] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[12]),
        .Q(async_path_bit[12]),
        .R(1'b0));
  FDRE \src_ff_reg[13] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[13]),
        .Q(async_path_bit[13]),
        .R(1'b0));
  FDRE \src_ff_reg[14] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[14]),
        .Q(async_path_bit[14]),
        .R(1'b0));
  FDRE \src_ff_reg[15] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[15]),
        .Q(async_path_bit[15]),
        .R(1'b0));
  FDRE \src_ff_reg[16] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[16]),
        .Q(async_path_bit[16]),
        .R(1'b0));
  FDRE \src_ff_reg[17] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[17]),
        .Q(async_path_bit[17]),
        .R(1'b0));
  FDRE \src_ff_reg[18] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[18]),
        .Q(async_path_bit[18]),
        .R(1'b0));
  FDRE \src_ff_reg[19] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[19]),
        .Q(async_path_bit[19]),
        .R(1'b0));
  FDRE \src_ff_reg[1] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[1]),
        .Q(async_path_bit[1]),
        .R(1'b0));
  FDRE \src_ff_reg[20] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[20]),
        .Q(async_path_bit[20]),
        .R(1'b0));
  FDRE \src_ff_reg[21] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[21]),
        .Q(async_path_bit[21]),
        .R(1'b0));
  FDRE \src_ff_reg[22] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[22]),
        .Q(async_path_bit[22]),
        .R(1'b0));
  FDRE \src_ff_reg[23] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[23]),
        .Q(async_path_bit[23]),
        .R(1'b0));
  FDRE \src_ff_reg[24] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[24]),
        .Q(async_path_bit[24]),
        .R(1'b0));
  FDRE \src_ff_reg[25] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[25]),
        .Q(async_path_bit[25]),
        .R(1'b0));
  FDRE \src_ff_reg[26] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[26]),
        .Q(async_path_bit[26]),
        .R(1'b0));
  FDRE \src_ff_reg[27] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[27]),
        .Q(async_path_bit[27]),
        .R(1'b0));
  FDRE \src_ff_reg[28] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[28]),
        .Q(async_path_bit[28]),
        .R(1'b0));
  FDRE \src_ff_reg[29] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[29]),
        .Q(async_path_bit[29]),
        .R(1'b0));
  FDRE \src_ff_reg[2] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[2]),
        .Q(async_path_bit[2]),
        .R(1'b0));
  FDRE \src_ff_reg[30] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[30]),
        .Q(async_path_bit[30]),
        .R(1'b0));
  FDRE \src_ff_reg[31] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[31]),
        .Q(async_path_bit[31]),
        .R(1'b0));
  FDRE \src_ff_reg[32] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[32]),
        .Q(async_path_bit[32]),
        .R(1'b0));
  FDRE \src_ff_reg[33] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[33]),
        .Q(async_path_bit[33]),
        .R(1'b0));
  FDRE \src_ff_reg[34] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[34]),
        .Q(async_path_bit[34]),
        .R(1'b0));
  FDRE \src_ff_reg[35] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[35]),
        .Q(async_path_bit[35]),
        .R(1'b0));
  FDRE \src_ff_reg[36] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[36]),
        .Q(async_path_bit[36]),
        .R(1'b0));
  FDRE \src_ff_reg[37] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[37]),
        .Q(async_path_bit[37]),
        .R(1'b0));
  FDRE \src_ff_reg[38] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[38]),
        .Q(async_path_bit[38]),
        .R(1'b0));
  FDRE \src_ff_reg[39] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[39]),
        .Q(async_path_bit[39]),
        .R(1'b0));
  FDRE \src_ff_reg[3] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[3]),
        .Q(async_path_bit[3]),
        .R(1'b0));
  FDRE \src_ff_reg[40] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[40]),
        .Q(async_path_bit[40]),
        .R(1'b0));
  FDRE \src_ff_reg[41] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[41]),
        .Q(async_path_bit[41]),
        .R(1'b0));
  FDRE \src_ff_reg[42] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[42]),
        .Q(async_path_bit[42]),
        .R(1'b0));
  FDRE \src_ff_reg[43] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[43]),
        .Q(async_path_bit[43]),
        .R(1'b0));
  FDRE \src_ff_reg[44] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[44]),
        .Q(async_path_bit[44]),
        .R(1'b0));
  FDRE \src_ff_reg[45] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[45]),
        .Q(async_path_bit[45]),
        .R(1'b0));
  FDRE \src_ff_reg[46] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[46]),
        .Q(async_path_bit[46]),
        .R(1'b0));
  FDRE \src_ff_reg[47] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[47]),
        .Q(async_path_bit[47]),
        .R(1'b0));
  FDRE \src_ff_reg[48] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[48]),
        .Q(async_path_bit[48]),
        .R(1'b0));
  FDRE \src_ff_reg[49] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[49]),
        .Q(async_path_bit[49]),
        .R(1'b0));
  FDRE \src_ff_reg[4] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[4]),
        .Q(async_path_bit[4]),
        .R(1'b0));
  FDRE \src_ff_reg[50] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[50]),
        .Q(async_path_bit[50]),
        .R(1'b0));
  FDRE \src_ff_reg[51] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[51]),
        .Q(async_path_bit[51]),
        .R(1'b0));
  FDRE \src_ff_reg[52] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[52]),
        .Q(async_path_bit[52]),
        .R(1'b0));
  FDRE \src_ff_reg[53] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[53]),
        .Q(async_path_bit[53]),
        .R(1'b0));
  FDRE \src_ff_reg[54] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[54]),
        .Q(async_path_bit[54]),
        .R(1'b0));
  FDRE \src_ff_reg[55] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[55]),
        .Q(async_path_bit[55]),
        .R(1'b0));
  FDRE \src_ff_reg[56] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[56]),
        .Q(async_path_bit[56]),
        .R(1'b0));
  FDRE \src_ff_reg[57] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[57]),
        .Q(async_path_bit[57]),
        .R(1'b0));
  FDRE \src_ff_reg[58] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[58]),
        .Q(async_path_bit[58]),
        .R(1'b0));
  FDRE \src_ff_reg[59] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[59]),
        .Q(async_path_bit[59]),
        .R(1'b0));
  FDRE \src_ff_reg[5] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[5]),
        .Q(async_path_bit[5]),
        .R(1'b0));
  FDRE \src_ff_reg[60] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[60]),
        .Q(async_path_bit[60]),
        .R(1'b0));
  FDRE \src_ff_reg[61] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[61]),
        .Q(async_path_bit[61]),
        .R(1'b0));
  FDRE \src_ff_reg[62] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[62]),
        .Q(async_path_bit[62]),
        .R(1'b0));
  FDRE \src_ff_reg[63] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[63]),
        .Q(async_path_bit[63]),
        .R(1'b0));
  FDRE \src_ff_reg[6] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[6]),
        .Q(async_path_bit[6]),
        .R(1'b0));
  FDRE \src_ff_reg[7] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[7]),
        .Q(async_path_bit[7]),
        .R(1'b0));
  FDRE \src_ff_reg[8] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[8]),
        .Q(async_path_bit[8]),
        .R(1'b0));
  FDRE \src_ff_reg[9] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[9]),
        .Q(async_path_bit[9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[0]),
        .Q(\syncstages_ff[0] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[10]),
        .Q(\syncstages_ff[0] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[11]),
        .Q(\syncstages_ff[0] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[12]),
        .Q(\syncstages_ff[0] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[13]),
        .Q(\syncstages_ff[0] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[14]),
        .Q(\syncstages_ff[0] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[15]),
        .Q(\syncstages_ff[0] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[16]),
        .Q(\syncstages_ff[0] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[17]),
        .Q(\syncstages_ff[0] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[18]),
        .Q(\syncstages_ff[0] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[19]),
        .Q(\syncstages_ff[0] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[1]),
        .Q(\syncstages_ff[0] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[20]),
        .Q(\syncstages_ff[0] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[21]),
        .Q(\syncstages_ff[0] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[22]),
        .Q(\syncstages_ff[0] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[23]),
        .Q(\syncstages_ff[0] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[24]),
        .Q(\syncstages_ff[0] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[25]),
        .Q(\syncstages_ff[0] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[26]),
        .Q(\syncstages_ff[0] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[27]),
        .Q(\syncstages_ff[0] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[28]),
        .Q(\syncstages_ff[0] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[29]),
        .Q(\syncstages_ff[0] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[2]),
        .Q(\syncstages_ff[0] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[30]),
        .Q(\syncstages_ff[0] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[31]),
        .Q(\syncstages_ff[0] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][32] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[32]),
        .Q(\syncstages_ff[0] [32]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][33] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[33]),
        .Q(\syncstages_ff[0] [33]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][34] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[34]),
        .Q(\syncstages_ff[0] [34]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][35] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[35]),
        .Q(\syncstages_ff[0] [35]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][36] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[36]),
        .Q(\syncstages_ff[0] [36]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][37] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[37]),
        .Q(\syncstages_ff[0] [37]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][38] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[38]),
        .Q(\syncstages_ff[0] [38]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][39] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[39]),
        .Q(\syncstages_ff[0] [39]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[3]),
        .Q(\syncstages_ff[0] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][40] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[40]),
        .Q(\syncstages_ff[0] [40]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][41] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[41]),
        .Q(\syncstages_ff[0] [41]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][42] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[42]),
        .Q(\syncstages_ff[0] [42]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][43] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[43]),
        .Q(\syncstages_ff[0] [43]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][44] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[44]),
        .Q(\syncstages_ff[0] [44]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][45] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[45]),
        .Q(\syncstages_ff[0] [45]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][46] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[46]),
        .Q(\syncstages_ff[0] [46]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][47] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[47]),
        .Q(\syncstages_ff[0] [47]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][48] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[48]),
        .Q(\syncstages_ff[0] [48]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][49] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[49]),
        .Q(\syncstages_ff[0] [49]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[4]),
        .Q(\syncstages_ff[0] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][50] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[50]),
        .Q(\syncstages_ff[0] [50]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][51] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[51]),
        .Q(\syncstages_ff[0] [51]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][52] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[52]),
        .Q(\syncstages_ff[0] [52]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][53] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[53]),
        .Q(\syncstages_ff[0] [53]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][54] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[54]),
        .Q(\syncstages_ff[0] [54]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][55] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[55]),
        .Q(\syncstages_ff[0] [55]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][56] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[56]),
        .Q(\syncstages_ff[0] [56]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][57] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[57]),
        .Q(\syncstages_ff[0] [57]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][58] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[58]),
        .Q(\syncstages_ff[0] [58]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][59] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[59]),
        .Q(\syncstages_ff[0] [59]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[5]),
        .Q(\syncstages_ff[0] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][60] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[60]),
        .Q(\syncstages_ff[0] [60]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][61] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[61]),
        .Q(\syncstages_ff[0] [61]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][62] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[62]),
        .Q(\syncstages_ff[0] [62]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][63] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[63]),
        .Q(\syncstages_ff[0] [63]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[6]),
        .Q(\syncstages_ff[0] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[7]),
        .Q(\syncstages_ff[0] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[8]),
        .Q(\syncstages_ff[0] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[9]),
        .Q(\syncstages_ff[0] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [0]),
        .Q(\syncstages_ff[1] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [10]),
        .Q(\syncstages_ff[1] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [11]),
        .Q(\syncstages_ff[1] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [12]),
        .Q(\syncstages_ff[1] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [13]),
        .Q(\syncstages_ff[1] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [14]),
        .Q(\syncstages_ff[1] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [15]),
        .Q(\syncstages_ff[1] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [16]),
        .Q(\syncstages_ff[1] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [17]),
        .Q(\syncstages_ff[1] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [18]),
        .Q(\syncstages_ff[1] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [19]),
        .Q(\syncstages_ff[1] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [1]),
        .Q(\syncstages_ff[1] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [20]),
        .Q(\syncstages_ff[1] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [21]),
        .Q(\syncstages_ff[1] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [22]),
        .Q(\syncstages_ff[1] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [23]),
        .Q(\syncstages_ff[1] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [24]),
        .Q(\syncstages_ff[1] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [25]),
        .Q(\syncstages_ff[1] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [26]),
        .Q(\syncstages_ff[1] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [27]),
        .Q(\syncstages_ff[1] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [28]),
        .Q(\syncstages_ff[1] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [29]),
        .Q(\syncstages_ff[1] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [2]),
        .Q(\syncstages_ff[1] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [30]),
        .Q(\syncstages_ff[1] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [31]),
        .Q(\syncstages_ff[1] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][32] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [32]),
        .Q(\syncstages_ff[1] [32]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][33] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [33]),
        .Q(\syncstages_ff[1] [33]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][34] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [34]),
        .Q(\syncstages_ff[1] [34]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][35] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [35]),
        .Q(\syncstages_ff[1] [35]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][36] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [36]),
        .Q(\syncstages_ff[1] [36]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][37] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [37]),
        .Q(\syncstages_ff[1] [37]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][38] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [38]),
        .Q(\syncstages_ff[1] [38]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][39] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [39]),
        .Q(\syncstages_ff[1] [39]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [3]),
        .Q(\syncstages_ff[1] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][40] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [40]),
        .Q(\syncstages_ff[1] [40]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][41] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [41]),
        .Q(\syncstages_ff[1] [41]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][42] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [42]),
        .Q(\syncstages_ff[1] [42]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][43] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [43]),
        .Q(\syncstages_ff[1] [43]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][44] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [44]),
        .Q(\syncstages_ff[1] [44]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][45] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [45]),
        .Q(\syncstages_ff[1] [45]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][46] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [46]),
        .Q(\syncstages_ff[1] [46]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][47] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [47]),
        .Q(\syncstages_ff[1] [47]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][48] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [48]),
        .Q(\syncstages_ff[1] [48]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][49] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [49]),
        .Q(\syncstages_ff[1] [49]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [4]),
        .Q(\syncstages_ff[1] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][50] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [50]),
        .Q(\syncstages_ff[1] [50]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][51] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [51]),
        .Q(\syncstages_ff[1] [51]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][52] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [52]),
        .Q(\syncstages_ff[1] [52]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][53] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [53]),
        .Q(\syncstages_ff[1] [53]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][54] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [54]),
        .Q(\syncstages_ff[1] [54]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][55] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [55]),
        .Q(\syncstages_ff[1] [55]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][56] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [56]),
        .Q(\syncstages_ff[1] [56]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][57] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [57]),
        .Q(\syncstages_ff[1] [57]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][58] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [58]),
        .Q(\syncstages_ff[1] [58]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][59] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [59]),
        .Q(\syncstages_ff[1] [59]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [5]),
        .Q(\syncstages_ff[1] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][60] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [60]),
        .Q(\syncstages_ff[1] [60]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][61] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [61]),
        .Q(\syncstages_ff[1] [61]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][62] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [62]),
        .Q(\syncstages_ff[1] [62]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][63] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [63]),
        .Q(\syncstages_ff[1] [63]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [6]),
        .Q(\syncstages_ff[1] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [7]),
        .Q(\syncstages_ff[1] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [8]),
        .Q(\syncstages_ff[1] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [9]),
        .Q(\syncstages_ff[1] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [0]),
        .Q(\syncstages_ff[2] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [10]),
        .Q(\syncstages_ff[2] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [11]),
        .Q(\syncstages_ff[2] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [12]),
        .Q(\syncstages_ff[2] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [13]),
        .Q(\syncstages_ff[2] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [14]),
        .Q(\syncstages_ff[2] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [15]),
        .Q(\syncstages_ff[2] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [16]),
        .Q(\syncstages_ff[2] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [17]),
        .Q(\syncstages_ff[2] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [18]),
        .Q(\syncstages_ff[2] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [19]),
        .Q(\syncstages_ff[2] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [1]),
        .Q(\syncstages_ff[2] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [20]),
        .Q(\syncstages_ff[2] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [21]),
        .Q(\syncstages_ff[2] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [22]),
        .Q(\syncstages_ff[2] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [23]),
        .Q(\syncstages_ff[2] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [24]),
        .Q(\syncstages_ff[2] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [25]),
        .Q(\syncstages_ff[2] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [26]),
        .Q(\syncstages_ff[2] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [27]),
        .Q(\syncstages_ff[2] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [28]),
        .Q(\syncstages_ff[2] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [29]),
        .Q(\syncstages_ff[2] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [2]),
        .Q(\syncstages_ff[2] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [30]),
        .Q(\syncstages_ff[2] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [31]),
        .Q(\syncstages_ff[2] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][32] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [32]),
        .Q(\syncstages_ff[2] [32]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][33] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [33]),
        .Q(\syncstages_ff[2] [33]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][34] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [34]),
        .Q(\syncstages_ff[2] [34]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][35] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [35]),
        .Q(\syncstages_ff[2] [35]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][36] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [36]),
        .Q(\syncstages_ff[2] [36]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][37] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [37]),
        .Q(\syncstages_ff[2] [37]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][38] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [38]),
        .Q(\syncstages_ff[2] [38]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][39] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [39]),
        .Q(\syncstages_ff[2] [39]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [3]),
        .Q(\syncstages_ff[2] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][40] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [40]),
        .Q(\syncstages_ff[2] [40]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][41] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [41]),
        .Q(\syncstages_ff[2] [41]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][42] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [42]),
        .Q(\syncstages_ff[2] [42]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][43] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [43]),
        .Q(\syncstages_ff[2] [43]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][44] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [44]),
        .Q(\syncstages_ff[2] [44]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][45] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [45]),
        .Q(\syncstages_ff[2] [45]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][46] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [46]),
        .Q(\syncstages_ff[2] [46]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][47] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [47]),
        .Q(\syncstages_ff[2] [47]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][48] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [48]),
        .Q(\syncstages_ff[2] [48]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][49] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [49]),
        .Q(\syncstages_ff[2] [49]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [4]),
        .Q(\syncstages_ff[2] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][50] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [50]),
        .Q(\syncstages_ff[2] [50]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][51] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [51]),
        .Q(\syncstages_ff[2] [51]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][52] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [52]),
        .Q(\syncstages_ff[2] [52]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][53] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [53]),
        .Q(\syncstages_ff[2] [53]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][54] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [54]),
        .Q(\syncstages_ff[2] [54]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][55] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [55]),
        .Q(\syncstages_ff[2] [55]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][56] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [56]),
        .Q(\syncstages_ff[2] [56]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][57] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [57]),
        .Q(\syncstages_ff[2] [57]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][58] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [58]),
        .Q(\syncstages_ff[2] [58]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][59] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [59]),
        .Q(\syncstages_ff[2] [59]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [5]),
        .Q(\syncstages_ff[2] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][60] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [60]),
        .Q(\syncstages_ff[2] [60]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][61] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [61]),
        .Q(\syncstages_ff[2] [61]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][62] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [62]),
        .Q(\syncstages_ff[2] [62]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][63] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [63]),
        .Q(\syncstages_ff[2] [63]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [6]),
        .Q(\syncstages_ff[2] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [7]),
        .Q(\syncstages_ff[2] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [8]),
        .Q(\syncstages_ff[2] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [9]),
        .Q(\syncstages_ff[2] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [0]),
        .Q(\syncstages_ff[3] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [10]),
        .Q(\syncstages_ff[3] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [11]),
        .Q(\syncstages_ff[3] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [12]),
        .Q(\syncstages_ff[3] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [13]),
        .Q(\syncstages_ff[3] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [14]),
        .Q(\syncstages_ff[3] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [15]),
        .Q(\syncstages_ff[3] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [16]),
        .Q(\syncstages_ff[3] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [17]),
        .Q(\syncstages_ff[3] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [18]),
        .Q(\syncstages_ff[3] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [19]),
        .Q(\syncstages_ff[3] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [1]),
        .Q(\syncstages_ff[3] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [20]),
        .Q(\syncstages_ff[3] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [21]),
        .Q(\syncstages_ff[3] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [22]),
        .Q(\syncstages_ff[3] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [23]),
        .Q(\syncstages_ff[3] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [24]),
        .Q(\syncstages_ff[3] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [25]),
        .Q(\syncstages_ff[3] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [26]),
        .Q(\syncstages_ff[3] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [27]),
        .Q(\syncstages_ff[3] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [28]),
        .Q(\syncstages_ff[3] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [29]),
        .Q(\syncstages_ff[3] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [2]),
        .Q(\syncstages_ff[3] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [30]),
        .Q(\syncstages_ff[3] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [31]),
        .Q(\syncstages_ff[3] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][32] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [32]),
        .Q(\syncstages_ff[3] [32]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][33] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [33]),
        .Q(\syncstages_ff[3] [33]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][34] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [34]),
        .Q(\syncstages_ff[3] [34]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][35] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [35]),
        .Q(\syncstages_ff[3] [35]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][36] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [36]),
        .Q(\syncstages_ff[3] [36]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][37] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [37]),
        .Q(\syncstages_ff[3] [37]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][38] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [38]),
        .Q(\syncstages_ff[3] [38]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][39] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [39]),
        .Q(\syncstages_ff[3] [39]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [3]),
        .Q(\syncstages_ff[3] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][40] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [40]),
        .Q(\syncstages_ff[3] [40]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][41] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [41]),
        .Q(\syncstages_ff[3] [41]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][42] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [42]),
        .Q(\syncstages_ff[3] [42]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][43] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [43]),
        .Q(\syncstages_ff[3] [43]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][44] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [44]),
        .Q(\syncstages_ff[3] [44]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][45] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [45]),
        .Q(\syncstages_ff[3] [45]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][46] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [46]),
        .Q(\syncstages_ff[3] [46]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][47] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [47]),
        .Q(\syncstages_ff[3] [47]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][48] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [48]),
        .Q(\syncstages_ff[3] [48]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][49] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [49]),
        .Q(\syncstages_ff[3] [49]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [4]),
        .Q(\syncstages_ff[3] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][50] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [50]),
        .Q(\syncstages_ff[3] [50]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][51] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [51]),
        .Q(\syncstages_ff[3] [51]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][52] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [52]),
        .Q(\syncstages_ff[3] [52]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][53] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [53]),
        .Q(\syncstages_ff[3] [53]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][54] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [54]),
        .Q(\syncstages_ff[3] [54]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][55] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [55]),
        .Q(\syncstages_ff[3] [55]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][56] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [56]),
        .Q(\syncstages_ff[3] [56]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][57] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [57]),
        .Q(\syncstages_ff[3] [57]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][58] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [58]),
        .Q(\syncstages_ff[3] [58]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][59] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [59]),
        .Q(\syncstages_ff[3] [59]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [5]),
        .Q(\syncstages_ff[3] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][60] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [60]),
        .Q(\syncstages_ff[3] [60]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][61] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [61]),
        .Q(\syncstages_ff[3] [61]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][62] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [62]),
        .Q(\syncstages_ff[3] [62]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][63] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [63]),
        .Q(\syncstages_ff[3] [63]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [6]),
        .Q(\syncstages_ff[3] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [7]),
        .Q(\syncstages_ff[3] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [8]),
        .Q(\syncstages_ff[3] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [9]),
        .Q(\syncstages_ff[3] [9]),
        .R(1'b0));
endmodule

(* DEST_SYNC_FF = "4" *) (* INIT_SYNC_FF = "0" *) (* ORIG_REF_NAME = "xpm_cdc_array_single" *) 
(* SIM_ASSERT_CHK = "0" *) (* SRC_INPUT_REG = "1" *) (* VERSION = "0" *) 
(* WIDTH = "64" *) (* XPM_MODULE = "TRUE" *) (* xpm_cdc = "ARRAY_SINGLE" *) 
module xpm_cdc_array_single__parameterized1__2
   (src_clk,
    src_in,
    dest_clk,
    dest_out);
  input src_clk;
  input [63:0]src_in;
  input dest_clk;
  output [63:0]dest_out;

  wire [63:0]async_path_bit;
  wire dest_clk;
  wire src_clk;
  wire [63:0]src_in;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [63:0]\syncstages_ff[0] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [63:0]\syncstages_ff[1] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [63:0]\syncstages_ff[2] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [63:0]\syncstages_ff[3] ;

  assign dest_out[63:0] = \syncstages_ff[3] ;
  FDRE \src_ff_reg[0] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[0]),
        .Q(async_path_bit[0]),
        .R(1'b0));
  FDRE \src_ff_reg[10] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[10]),
        .Q(async_path_bit[10]),
        .R(1'b0));
  FDRE \src_ff_reg[11] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[11]),
        .Q(async_path_bit[11]),
        .R(1'b0));
  FDRE \src_ff_reg[12] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[12]),
        .Q(async_path_bit[12]),
        .R(1'b0));
  FDRE \src_ff_reg[13] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[13]),
        .Q(async_path_bit[13]),
        .R(1'b0));
  FDRE \src_ff_reg[14] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[14]),
        .Q(async_path_bit[14]),
        .R(1'b0));
  FDRE \src_ff_reg[15] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[15]),
        .Q(async_path_bit[15]),
        .R(1'b0));
  FDRE \src_ff_reg[16] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[16]),
        .Q(async_path_bit[16]),
        .R(1'b0));
  FDRE \src_ff_reg[17] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[17]),
        .Q(async_path_bit[17]),
        .R(1'b0));
  FDRE \src_ff_reg[18] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[18]),
        .Q(async_path_bit[18]),
        .R(1'b0));
  FDRE \src_ff_reg[19] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[19]),
        .Q(async_path_bit[19]),
        .R(1'b0));
  FDRE \src_ff_reg[1] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[1]),
        .Q(async_path_bit[1]),
        .R(1'b0));
  FDRE \src_ff_reg[20] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[20]),
        .Q(async_path_bit[20]),
        .R(1'b0));
  FDRE \src_ff_reg[21] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[21]),
        .Q(async_path_bit[21]),
        .R(1'b0));
  FDRE \src_ff_reg[22] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[22]),
        .Q(async_path_bit[22]),
        .R(1'b0));
  FDRE \src_ff_reg[23] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[23]),
        .Q(async_path_bit[23]),
        .R(1'b0));
  FDRE \src_ff_reg[24] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[24]),
        .Q(async_path_bit[24]),
        .R(1'b0));
  FDRE \src_ff_reg[25] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[25]),
        .Q(async_path_bit[25]),
        .R(1'b0));
  FDRE \src_ff_reg[26] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[26]),
        .Q(async_path_bit[26]),
        .R(1'b0));
  FDRE \src_ff_reg[27] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[27]),
        .Q(async_path_bit[27]),
        .R(1'b0));
  FDRE \src_ff_reg[28] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[28]),
        .Q(async_path_bit[28]),
        .R(1'b0));
  FDRE \src_ff_reg[29] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[29]),
        .Q(async_path_bit[29]),
        .R(1'b0));
  FDRE \src_ff_reg[2] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[2]),
        .Q(async_path_bit[2]),
        .R(1'b0));
  FDRE \src_ff_reg[30] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[30]),
        .Q(async_path_bit[30]),
        .R(1'b0));
  FDRE \src_ff_reg[31] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[31]),
        .Q(async_path_bit[31]),
        .R(1'b0));
  FDRE \src_ff_reg[32] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[32]),
        .Q(async_path_bit[32]),
        .R(1'b0));
  FDRE \src_ff_reg[33] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[33]),
        .Q(async_path_bit[33]),
        .R(1'b0));
  FDRE \src_ff_reg[34] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[34]),
        .Q(async_path_bit[34]),
        .R(1'b0));
  FDRE \src_ff_reg[35] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[35]),
        .Q(async_path_bit[35]),
        .R(1'b0));
  FDRE \src_ff_reg[36] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[36]),
        .Q(async_path_bit[36]),
        .R(1'b0));
  FDRE \src_ff_reg[37] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[37]),
        .Q(async_path_bit[37]),
        .R(1'b0));
  FDRE \src_ff_reg[38] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[38]),
        .Q(async_path_bit[38]),
        .R(1'b0));
  FDRE \src_ff_reg[39] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[39]),
        .Q(async_path_bit[39]),
        .R(1'b0));
  FDRE \src_ff_reg[3] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[3]),
        .Q(async_path_bit[3]),
        .R(1'b0));
  FDRE \src_ff_reg[40] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[40]),
        .Q(async_path_bit[40]),
        .R(1'b0));
  FDRE \src_ff_reg[41] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[41]),
        .Q(async_path_bit[41]),
        .R(1'b0));
  FDRE \src_ff_reg[42] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[42]),
        .Q(async_path_bit[42]),
        .R(1'b0));
  FDRE \src_ff_reg[43] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[43]),
        .Q(async_path_bit[43]),
        .R(1'b0));
  FDRE \src_ff_reg[44] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[44]),
        .Q(async_path_bit[44]),
        .R(1'b0));
  FDRE \src_ff_reg[45] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[45]),
        .Q(async_path_bit[45]),
        .R(1'b0));
  FDRE \src_ff_reg[46] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[46]),
        .Q(async_path_bit[46]),
        .R(1'b0));
  FDRE \src_ff_reg[47] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[47]),
        .Q(async_path_bit[47]),
        .R(1'b0));
  FDRE \src_ff_reg[48] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[48]),
        .Q(async_path_bit[48]),
        .R(1'b0));
  FDRE \src_ff_reg[49] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[49]),
        .Q(async_path_bit[49]),
        .R(1'b0));
  FDRE \src_ff_reg[4] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[4]),
        .Q(async_path_bit[4]),
        .R(1'b0));
  FDRE \src_ff_reg[50] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[50]),
        .Q(async_path_bit[50]),
        .R(1'b0));
  FDRE \src_ff_reg[51] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[51]),
        .Q(async_path_bit[51]),
        .R(1'b0));
  FDRE \src_ff_reg[52] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[52]),
        .Q(async_path_bit[52]),
        .R(1'b0));
  FDRE \src_ff_reg[53] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[53]),
        .Q(async_path_bit[53]),
        .R(1'b0));
  FDRE \src_ff_reg[54] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[54]),
        .Q(async_path_bit[54]),
        .R(1'b0));
  FDRE \src_ff_reg[55] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[55]),
        .Q(async_path_bit[55]),
        .R(1'b0));
  FDRE \src_ff_reg[56] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[56]),
        .Q(async_path_bit[56]),
        .R(1'b0));
  FDRE \src_ff_reg[57] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[57]),
        .Q(async_path_bit[57]),
        .R(1'b0));
  FDRE \src_ff_reg[58] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[58]),
        .Q(async_path_bit[58]),
        .R(1'b0));
  FDRE \src_ff_reg[59] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[59]),
        .Q(async_path_bit[59]),
        .R(1'b0));
  FDRE \src_ff_reg[5] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[5]),
        .Q(async_path_bit[5]),
        .R(1'b0));
  FDRE \src_ff_reg[60] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[60]),
        .Q(async_path_bit[60]),
        .R(1'b0));
  FDRE \src_ff_reg[61] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[61]),
        .Q(async_path_bit[61]),
        .R(1'b0));
  FDRE \src_ff_reg[62] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[62]),
        .Q(async_path_bit[62]),
        .R(1'b0));
  FDRE \src_ff_reg[63] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[63]),
        .Q(async_path_bit[63]),
        .R(1'b0));
  FDRE \src_ff_reg[6] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[6]),
        .Q(async_path_bit[6]),
        .R(1'b0));
  FDRE \src_ff_reg[7] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[7]),
        .Q(async_path_bit[7]),
        .R(1'b0));
  FDRE \src_ff_reg[8] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[8]),
        .Q(async_path_bit[8]),
        .R(1'b0));
  FDRE \src_ff_reg[9] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[9]),
        .Q(async_path_bit[9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[0]),
        .Q(\syncstages_ff[0] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[10]),
        .Q(\syncstages_ff[0] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[11]),
        .Q(\syncstages_ff[0] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[12]),
        .Q(\syncstages_ff[0] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[13]),
        .Q(\syncstages_ff[0] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[14]),
        .Q(\syncstages_ff[0] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[15]),
        .Q(\syncstages_ff[0] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[16]),
        .Q(\syncstages_ff[0] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[17]),
        .Q(\syncstages_ff[0] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[18]),
        .Q(\syncstages_ff[0] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[19]),
        .Q(\syncstages_ff[0] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[1]),
        .Q(\syncstages_ff[0] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[20]),
        .Q(\syncstages_ff[0] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[21]),
        .Q(\syncstages_ff[0] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[22]),
        .Q(\syncstages_ff[0] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[23]),
        .Q(\syncstages_ff[0] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[24]),
        .Q(\syncstages_ff[0] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[25]),
        .Q(\syncstages_ff[0] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[26]),
        .Q(\syncstages_ff[0] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[27]),
        .Q(\syncstages_ff[0] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[28]),
        .Q(\syncstages_ff[0] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[29]),
        .Q(\syncstages_ff[0] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[2]),
        .Q(\syncstages_ff[0] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[30]),
        .Q(\syncstages_ff[0] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[31]),
        .Q(\syncstages_ff[0] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][32] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[32]),
        .Q(\syncstages_ff[0] [32]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][33] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[33]),
        .Q(\syncstages_ff[0] [33]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][34] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[34]),
        .Q(\syncstages_ff[0] [34]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][35] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[35]),
        .Q(\syncstages_ff[0] [35]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][36] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[36]),
        .Q(\syncstages_ff[0] [36]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][37] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[37]),
        .Q(\syncstages_ff[0] [37]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][38] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[38]),
        .Q(\syncstages_ff[0] [38]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][39] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[39]),
        .Q(\syncstages_ff[0] [39]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[3]),
        .Q(\syncstages_ff[0] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][40] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[40]),
        .Q(\syncstages_ff[0] [40]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][41] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[41]),
        .Q(\syncstages_ff[0] [41]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][42] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[42]),
        .Q(\syncstages_ff[0] [42]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][43] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[43]),
        .Q(\syncstages_ff[0] [43]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][44] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[44]),
        .Q(\syncstages_ff[0] [44]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][45] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[45]),
        .Q(\syncstages_ff[0] [45]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][46] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[46]),
        .Q(\syncstages_ff[0] [46]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][47] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[47]),
        .Q(\syncstages_ff[0] [47]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][48] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[48]),
        .Q(\syncstages_ff[0] [48]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][49] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[49]),
        .Q(\syncstages_ff[0] [49]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[4]),
        .Q(\syncstages_ff[0] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][50] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[50]),
        .Q(\syncstages_ff[0] [50]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][51] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[51]),
        .Q(\syncstages_ff[0] [51]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][52] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[52]),
        .Q(\syncstages_ff[0] [52]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][53] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[53]),
        .Q(\syncstages_ff[0] [53]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][54] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[54]),
        .Q(\syncstages_ff[0] [54]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][55] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[55]),
        .Q(\syncstages_ff[0] [55]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][56] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[56]),
        .Q(\syncstages_ff[0] [56]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][57] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[57]),
        .Q(\syncstages_ff[0] [57]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][58] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[58]),
        .Q(\syncstages_ff[0] [58]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][59] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[59]),
        .Q(\syncstages_ff[0] [59]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[5]),
        .Q(\syncstages_ff[0] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][60] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[60]),
        .Q(\syncstages_ff[0] [60]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][61] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[61]),
        .Q(\syncstages_ff[0] [61]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][62] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[62]),
        .Q(\syncstages_ff[0] [62]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][63] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[63]),
        .Q(\syncstages_ff[0] [63]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[6]),
        .Q(\syncstages_ff[0] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[7]),
        .Q(\syncstages_ff[0] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[8]),
        .Q(\syncstages_ff[0] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[9]),
        .Q(\syncstages_ff[0] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [0]),
        .Q(\syncstages_ff[1] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [10]),
        .Q(\syncstages_ff[1] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [11]),
        .Q(\syncstages_ff[1] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [12]),
        .Q(\syncstages_ff[1] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [13]),
        .Q(\syncstages_ff[1] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [14]),
        .Q(\syncstages_ff[1] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [15]),
        .Q(\syncstages_ff[1] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [16]),
        .Q(\syncstages_ff[1] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [17]),
        .Q(\syncstages_ff[1] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [18]),
        .Q(\syncstages_ff[1] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [19]),
        .Q(\syncstages_ff[1] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [1]),
        .Q(\syncstages_ff[1] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [20]),
        .Q(\syncstages_ff[1] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [21]),
        .Q(\syncstages_ff[1] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [22]),
        .Q(\syncstages_ff[1] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [23]),
        .Q(\syncstages_ff[1] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [24]),
        .Q(\syncstages_ff[1] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [25]),
        .Q(\syncstages_ff[1] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [26]),
        .Q(\syncstages_ff[1] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [27]),
        .Q(\syncstages_ff[1] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [28]),
        .Q(\syncstages_ff[1] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [29]),
        .Q(\syncstages_ff[1] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [2]),
        .Q(\syncstages_ff[1] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [30]),
        .Q(\syncstages_ff[1] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [31]),
        .Q(\syncstages_ff[1] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][32] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [32]),
        .Q(\syncstages_ff[1] [32]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][33] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [33]),
        .Q(\syncstages_ff[1] [33]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][34] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [34]),
        .Q(\syncstages_ff[1] [34]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][35] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [35]),
        .Q(\syncstages_ff[1] [35]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][36] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [36]),
        .Q(\syncstages_ff[1] [36]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][37] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [37]),
        .Q(\syncstages_ff[1] [37]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][38] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [38]),
        .Q(\syncstages_ff[1] [38]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][39] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [39]),
        .Q(\syncstages_ff[1] [39]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [3]),
        .Q(\syncstages_ff[1] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][40] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [40]),
        .Q(\syncstages_ff[1] [40]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][41] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [41]),
        .Q(\syncstages_ff[1] [41]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][42] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [42]),
        .Q(\syncstages_ff[1] [42]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][43] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [43]),
        .Q(\syncstages_ff[1] [43]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][44] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [44]),
        .Q(\syncstages_ff[1] [44]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][45] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [45]),
        .Q(\syncstages_ff[1] [45]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][46] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [46]),
        .Q(\syncstages_ff[1] [46]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][47] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [47]),
        .Q(\syncstages_ff[1] [47]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][48] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [48]),
        .Q(\syncstages_ff[1] [48]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][49] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [49]),
        .Q(\syncstages_ff[1] [49]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [4]),
        .Q(\syncstages_ff[1] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][50] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [50]),
        .Q(\syncstages_ff[1] [50]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][51] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [51]),
        .Q(\syncstages_ff[1] [51]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][52] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [52]),
        .Q(\syncstages_ff[1] [52]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][53] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [53]),
        .Q(\syncstages_ff[1] [53]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][54] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [54]),
        .Q(\syncstages_ff[1] [54]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][55] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [55]),
        .Q(\syncstages_ff[1] [55]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][56] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [56]),
        .Q(\syncstages_ff[1] [56]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][57] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [57]),
        .Q(\syncstages_ff[1] [57]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][58] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [58]),
        .Q(\syncstages_ff[1] [58]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][59] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [59]),
        .Q(\syncstages_ff[1] [59]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [5]),
        .Q(\syncstages_ff[1] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][60] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [60]),
        .Q(\syncstages_ff[1] [60]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][61] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [61]),
        .Q(\syncstages_ff[1] [61]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][62] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [62]),
        .Q(\syncstages_ff[1] [62]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][63] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [63]),
        .Q(\syncstages_ff[1] [63]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [6]),
        .Q(\syncstages_ff[1] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [7]),
        .Q(\syncstages_ff[1] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [8]),
        .Q(\syncstages_ff[1] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [9]),
        .Q(\syncstages_ff[1] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [0]),
        .Q(\syncstages_ff[2] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [10]),
        .Q(\syncstages_ff[2] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [11]),
        .Q(\syncstages_ff[2] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [12]),
        .Q(\syncstages_ff[2] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [13]),
        .Q(\syncstages_ff[2] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [14]),
        .Q(\syncstages_ff[2] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [15]),
        .Q(\syncstages_ff[2] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [16]),
        .Q(\syncstages_ff[2] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [17]),
        .Q(\syncstages_ff[2] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [18]),
        .Q(\syncstages_ff[2] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [19]),
        .Q(\syncstages_ff[2] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [1]),
        .Q(\syncstages_ff[2] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [20]),
        .Q(\syncstages_ff[2] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [21]),
        .Q(\syncstages_ff[2] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [22]),
        .Q(\syncstages_ff[2] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [23]),
        .Q(\syncstages_ff[2] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [24]),
        .Q(\syncstages_ff[2] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [25]),
        .Q(\syncstages_ff[2] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [26]),
        .Q(\syncstages_ff[2] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [27]),
        .Q(\syncstages_ff[2] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [28]),
        .Q(\syncstages_ff[2] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [29]),
        .Q(\syncstages_ff[2] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [2]),
        .Q(\syncstages_ff[2] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [30]),
        .Q(\syncstages_ff[2] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [31]),
        .Q(\syncstages_ff[2] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][32] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [32]),
        .Q(\syncstages_ff[2] [32]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][33] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [33]),
        .Q(\syncstages_ff[2] [33]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][34] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [34]),
        .Q(\syncstages_ff[2] [34]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][35] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [35]),
        .Q(\syncstages_ff[2] [35]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][36] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [36]),
        .Q(\syncstages_ff[2] [36]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][37] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [37]),
        .Q(\syncstages_ff[2] [37]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][38] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [38]),
        .Q(\syncstages_ff[2] [38]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][39] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [39]),
        .Q(\syncstages_ff[2] [39]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [3]),
        .Q(\syncstages_ff[2] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][40] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [40]),
        .Q(\syncstages_ff[2] [40]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][41] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [41]),
        .Q(\syncstages_ff[2] [41]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][42] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [42]),
        .Q(\syncstages_ff[2] [42]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][43] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [43]),
        .Q(\syncstages_ff[2] [43]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][44] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [44]),
        .Q(\syncstages_ff[2] [44]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][45] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [45]),
        .Q(\syncstages_ff[2] [45]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][46] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [46]),
        .Q(\syncstages_ff[2] [46]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][47] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [47]),
        .Q(\syncstages_ff[2] [47]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][48] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [48]),
        .Q(\syncstages_ff[2] [48]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][49] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [49]),
        .Q(\syncstages_ff[2] [49]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [4]),
        .Q(\syncstages_ff[2] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][50] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [50]),
        .Q(\syncstages_ff[2] [50]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][51] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [51]),
        .Q(\syncstages_ff[2] [51]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][52] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [52]),
        .Q(\syncstages_ff[2] [52]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][53] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [53]),
        .Q(\syncstages_ff[2] [53]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][54] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [54]),
        .Q(\syncstages_ff[2] [54]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][55] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [55]),
        .Q(\syncstages_ff[2] [55]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][56] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [56]),
        .Q(\syncstages_ff[2] [56]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][57] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [57]),
        .Q(\syncstages_ff[2] [57]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][58] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [58]),
        .Q(\syncstages_ff[2] [58]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][59] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [59]),
        .Q(\syncstages_ff[2] [59]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [5]),
        .Q(\syncstages_ff[2] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][60] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [60]),
        .Q(\syncstages_ff[2] [60]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][61] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [61]),
        .Q(\syncstages_ff[2] [61]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][62] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [62]),
        .Q(\syncstages_ff[2] [62]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][63] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [63]),
        .Q(\syncstages_ff[2] [63]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [6]),
        .Q(\syncstages_ff[2] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [7]),
        .Q(\syncstages_ff[2] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [8]),
        .Q(\syncstages_ff[2] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [9]),
        .Q(\syncstages_ff[2] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [0]),
        .Q(\syncstages_ff[3] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [10]),
        .Q(\syncstages_ff[3] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [11]),
        .Q(\syncstages_ff[3] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [12]),
        .Q(\syncstages_ff[3] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [13]),
        .Q(\syncstages_ff[3] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [14]),
        .Q(\syncstages_ff[3] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [15]),
        .Q(\syncstages_ff[3] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [16]),
        .Q(\syncstages_ff[3] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [17]),
        .Q(\syncstages_ff[3] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [18]),
        .Q(\syncstages_ff[3] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [19]),
        .Q(\syncstages_ff[3] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [1]),
        .Q(\syncstages_ff[3] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [20]),
        .Q(\syncstages_ff[3] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [21]),
        .Q(\syncstages_ff[3] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [22]),
        .Q(\syncstages_ff[3] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [23]),
        .Q(\syncstages_ff[3] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [24]),
        .Q(\syncstages_ff[3] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [25]),
        .Q(\syncstages_ff[3] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [26]),
        .Q(\syncstages_ff[3] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [27]),
        .Q(\syncstages_ff[3] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [28]),
        .Q(\syncstages_ff[3] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [29]),
        .Q(\syncstages_ff[3] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [2]),
        .Q(\syncstages_ff[3] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [30]),
        .Q(\syncstages_ff[3] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [31]),
        .Q(\syncstages_ff[3] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][32] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [32]),
        .Q(\syncstages_ff[3] [32]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][33] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [33]),
        .Q(\syncstages_ff[3] [33]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][34] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [34]),
        .Q(\syncstages_ff[3] [34]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][35] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [35]),
        .Q(\syncstages_ff[3] [35]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][36] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [36]),
        .Q(\syncstages_ff[3] [36]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][37] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [37]),
        .Q(\syncstages_ff[3] [37]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][38] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [38]),
        .Q(\syncstages_ff[3] [38]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][39] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [39]),
        .Q(\syncstages_ff[3] [39]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [3]),
        .Q(\syncstages_ff[3] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][40] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [40]),
        .Q(\syncstages_ff[3] [40]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][41] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [41]),
        .Q(\syncstages_ff[3] [41]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][42] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [42]),
        .Q(\syncstages_ff[3] [42]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][43] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [43]),
        .Q(\syncstages_ff[3] [43]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][44] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [44]),
        .Q(\syncstages_ff[3] [44]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][45] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [45]),
        .Q(\syncstages_ff[3] [45]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][46] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [46]),
        .Q(\syncstages_ff[3] [46]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][47] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [47]),
        .Q(\syncstages_ff[3] [47]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][48] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [48]),
        .Q(\syncstages_ff[3] [48]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][49] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [49]),
        .Q(\syncstages_ff[3] [49]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [4]),
        .Q(\syncstages_ff[3] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][50] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [50]),
        .Q(\syncstages_ff[3] [50]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][51] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [51]),
        .Q(\syncstages_ff[3] [51]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][52] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [52]),
        .Q(\syncstages_ff[3] [52]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][53] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [53]),
        .Q(\syncstages_ff[3] [53]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][54] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [54]),
        .Q(\syncstages_ff[3] [54]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][55] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [55]),
        .Q(\syncstages_ff[3] [55]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][56] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [56]),
        .Q(\syncstages_ff[3] [56]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][57] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [57]),
        .Q(\syncstages_ff[3] [57]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][58] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [58]),
        .Q(\syncstages_ff[3] [58]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][59] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [59]),
        .Q(\syncstages_ff[3] [59]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [5]),
        .Q(\syncstages_ff[3] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][60] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [60]),
        .Q(\syncstages_ff[3] [60]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][61] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [61]),
        .Q(\syncstages_ff[3] [61]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][62] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [62]),
        .Q(\syncstages_ff[3] [62]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][63] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [63]),
        .Q(\syncstages_ff[3] [63]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [6]),
        .Q(\syncstages_ff[3] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [7]),
        .Q(\syncstages_ff[3] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [8]),
        .Q(\syncstages_ff[3] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [9]),
        .Q(\syncstages_ff[3] [9]),
        .R(1'b0));
endmodule

(* DEST_SYNC_FF = "4" *) (* INIT_SYNC_FF = "0" *) (* ORIG_REF_NAME = "xpm_cdc_array_single" *) 
(* SIM_ASSERT_CHK = "0" *) (* SRC_INPUT_REG = "1" *) (* VERSION = "0" *) 
(* WIDTH = "96" *) (* XPM_MODULE = "TRUE" *) (* xpm_cdc = "ARRAY_SINGLE" *) 
module xpm_cdc_array_single__parameterized2
   (src_clk,
    src_in,
    dest_clk,
    dest_out);
  input src_clk;
  input [95:0]src_in;
  input dest_clk;
  output [95:0]dest_out;

  wire [95:0]async_path_bit;
  wire dest_clk;
  wire src_clk;
  wire [95:0]src_in;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [95:0]\syncstages_ff[0] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [95:0]\syncstages_ff[1] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [95:0]\syncstages_ff[2] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [95:0]\syncstages_ff[3] ;

  assign dest_out[95:0] = \syncstages_ff[3] ;
  FDRE \src_ff_reg[0] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[0]),
        .Q(async_path_bit[0]),
        .R(1'b0));
  FDRE \src_ff_reg[10] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[10]),
        .Q(async_path_bit[10]),
        .R(1'b0));
  FDRE \src_ff_reg[11] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[11]),
        .Q(async_path_bit[11]),
        .R(1'b0));
  FDRE \src_ff_reg[12] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[12]),
        .Q(async_path_bit[12]),
        .R(1'b0));
  FDRE \src_ff_reg[13] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[13]),
        .Q(async_path_bit[13]),
        .R(1'b0));
  FDRE \src_ff_reg[14] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[14]),
        .Q(async_path_bit[14]),
        .R(1'b0));
  FDRE \src_ff_reg[15] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[15]),
        .Q(async_path_bit[15]),
        .R(1'b0));
  FDRE \src_ff_reg[16] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[16]),
        .Q(async_path_bit[16]),
        .R(1'b0));
  FDRE \src_ff_reg[17] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[17]),
        .Q(async_path_bit[17]),
        .R(1'b0));
  FDRE \src_ff_reg[18] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[18]),
        .Q(async_path_bit[18]),
        .R(1'b0));
  FDRE \src_ff_reg[19] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[19]),
        .Q(async_path_bit[19]),
        .R(1'b0));
  FDRE \src_ff_reg[1] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[1]),
        .Q(async_path_bit[1]),
        .R(1'b0));
  FDRE \src_ff_reg[20] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[20]),
        .Q(async_path_bit[20]),
        .R(1'b0));
  FDRE \src_ff_reg[21] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[21]),
        .Q(async_path_bit[21]),
        .R(1'b0));
  FDRE \src_ff_reg[22] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[22]),
        .Q(async_path_bit[22]),
        .R(1'b0));
  FDRE \src_ff_reg[23] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[23]),
        .Q(async_path_bit[23]),
        .R(1'b0));
  FDRE \src_ff_reg[24] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[24]),
        .Q(async_path_bit[24]),
        .R(1'b0));
  FDRE \src_ff_reg[25] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[25]),
        .Q(async_path_bit[25]),
        .R(1'b0));
  FDRE \src_ff_reg[26] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[26]),
        .Q(async_path_bit[26]),
        .R(1'b0));
  FDRE \src_ff_reg[27] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[27]),
        .Q(async_path_bit[27]),
        .R(1'b0));
  FDRE \src_ff_reg[28] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[28]),
        .Q(async_path_bit[28]),
        .R(1'b0));
  FDRE \src_ff_reg[29] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[29]),
        .Q(async_path_bit[29]),
        .R(1'b0));
  FDRE \src_ff_reg[2] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[2]),
        .Q(async_path_bit[2]),
        .R(1'b0));
  FDRE \src_ff_reg[30] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[30]),
        .Q(async_path_bit[30]),
        .R(1'b0));
  FDRE \src_ff_reg[31] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[31]),
        .Q(async_path_bit[31]),
        .R(1'b0));
  FDRE \src_ff_reg[32] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[32]),
        .Q(async_path_bit[32]),
        .R(1'b0));
  FDRE \src_ff_reg[33] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[33]),
        .Q(async_path_bit[33]),
        .R(1'b0));
  FDRE \src_ff_reg[34] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[34]),
        .Q(async_path_bit[34]),
        .R(1'b0));
  FDRE \src_ff_reg[35] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[35]),
        .Q(async_path_bit[35]),
        .R(1'b0));
  FDRE \src_ff_reg[36] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[36]),
        .Q(async_path_bit[36]),
        .R(1'b0));
  FDRE \src_ff_reg[37] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[37]),
        .Q(async_path_bit[37]),
        .R(1'b0));
  FDRE \src_ff_reg[38] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[38]),
        .Q(async_path_bit[38]),
        .R(1'b0));
  FDRE \src_ff_reg[39] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[39]),
        .Q(async_path_bit[39]),
        .R(1'b0));
  FDRE \src_ff_reg[3] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[3]),
        .Q(async_path_bit[3]),
        .R(1'b0));
  FDRE \src_ff_reg[40] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[40]),
        .Q(async_path_bit[40]),
        .R(1'b0));
  FDRE \src_ff_reg[41] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[41]),
        .Q(async_path_bit[41]),
        .R(1'b0));
  FDRE \src_ff_reg[42] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[42]),
        .Q(async_path_bit[42]),
        .R(1'b0));
  FDRE \src_ff_reg[43] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[43]),
        .Q(async_path_bit[43]),
        .R(1'b0));
  FDRE \src_ff_reg[44] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[44]),
        .Q(async_path_bit[44]),
        .R(1'b0));
  FDRE \src_ff_reg[45] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[45]),
        .Q(async_path_bit[45]),
        .R(1'b0));
  FDRE \src_ff_reg[46] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[46]),
        .Q(async_path_bit[46]),
        .R(1'b0));
  FDRE \src_ff_reg[47] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[47]),
        .Q(async_path_bit[47]),
        .R(1'b0));
  FDRE \src_ff_reg[48] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[48]),
        .Q(async_path_bit[48]),
        .R(1'b0));
  FDRE \src_ff_reg[49] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[49]),
        .Q(async_path_bit[49]),
        .R(1'b0));
  FDRE \src_ff_reg[4] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[4]),
        .Q(async_path_bit[4]),
        .R(1'b0));
  FDRE \src_ff_reg[50] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[50]),
        .Q(async_path_bit[50]),
        .R(1'b0));
  FDRE \src_ff_reg[51] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[51]),
        .Q(async_path_bit[51]),
        .R(1'b0));
  FDRE \src_ff_reg[52] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[52]),
        .Q(async_path_bit[52]),
        .R(1'b0));
  FDRE \src_ff_reg[53] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[53]),
        .Q(async_path_bit[53]),
        .R(1'b0));
  FDRE \src_ff_reg[54] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[54]),
        .Q(async_path_bit[54]),
        .R(1'b0));
  FDRE \src_ff_reg[55] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[55]),
        .Q(async_path_bit[55]),
        .R(1'b0));
  FDRE \src_ff_reg[56] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[56]),
        .Q(async_path_bit[56]),
        .R(1'b0));
  FDRE \src_ff_reg[57] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[57]),
        .Q(async_path_bit[57]),
        .R(1'b0));
  FDRE \src_ff_reg[58] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[58]),
        .Q(async_path_bit[58]),
        .R(1'b0));
  FDRE \src_ff_reg[59] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[59]),
        .Q(async_path_bit[59]),
        .R(1'b0));
  FDRE \src_ff_reg[5] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[5]),
        .Q(async_path_bit[5]),
        .R(1'b0));
  FDRE \src_ff_reg[60] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[60]),
        .Q(async_path_bit[60]),
        .R(1'b0));
  FDRE \src_ff_reg[61] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[61]),
        .Q(async_path_bit[61]),
        .R(1'b0));
  FDRE \src_ff_reg[62] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[62]),
        .Q(async_path_bit[62]),
        .R(1'b0));
  FDRE \src_ff_reg[63] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[63]),
        .Q(async_path_bit[63]),
        .R(1'b0));
  FDRE \src_ff_reg[64] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[64]),
        .Q(async_path_bit[64]),
        .R(1'b0));
  FDRE \src_ff_reg[65] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[65]),
        .Q(async_path_bit[65]),
        .R(1'b0));
  FDRE \src_ff_reg[66] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[66]),
        .Q(async_path_bit[66]),
        .R(1'b0));
  FDRE \src_ff_reg[67] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[67]),
        .Q(async_path_bit[67]),
        .R(1'b0));
  FDRE \src_ff_reg[68] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[68]),
        .Q(async_path_bit[68]),
        .R(1'b0));
  FDRE \src_ff_reg[69] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[69]),
        .Q(async_path_bit[69]),
        .R(1'b0));
  FDRE \src_ff_reg[6] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[6]),
        .Q(async_path_bit[6]),
        .R(1'b0));
  FDRE \src_ff_reg[70] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[70]),
        .Q(async_path_bit[70]),
        .R(1'b0));
  FDRE \src_ff_reg[71] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[71]),
        .Q(async_path_bit[71]),
        .R(1'b0));
  FDRE \src_ff_reg[72] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[72]),
        .Q(async_path_bit[72]),
        .R(1'b0));
  FDRE \src_ff_reg[73] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[73]),
        .Q(async_path_bit[73]),
        .R(1'b0));
  FDRE \src_ff_reg[74] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[74]),
        .Q(async_path_bit[74]),
        .R(1'b0));
  FDRE \src_ff_reg[75] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[75]),
        .Q(async_path_bit[75]),
        .R(1'b0));
  FDRE \src_ff_reg[76] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[76]),
        .Q(async_path_bit[76]),
        .R(1'b0));
  FDRE \src_ff_reg[77] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[77]),
        .Q(async_path_bit[77]),
        .R(1'b0));
  FDRE \src_ff_reg[78] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[78]),
        .Q(async_path_bit[78]),
        .R(1'b0));
  FDRE \src_ff_reg[79] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[79]),
        .Q(async_path_bit[79]),
        .R(1'b0));
  FDRE \src_ff_reg[7] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[7]),
        .Q(async_path_bit[7]),
        .R(1'b0));
  FDRE \src_ff_reg[80] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[80]),
        .Q(async_path_bit[80]),
        .R(1'b0));
  FDRE \src_ff_reg[81] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[81]),
        .Q(async_path_bit[81]),
        .R(1'b0));
  FDRE \src_ff_reg[82] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[82]),
        .Q(async_path_bit[82]),
        .R(1'b0));
  FDRE \src_ff_reg[83] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[83]),
        .Q(async_path_bit[83]),
        .R(1'b0));
  FDRE \src_ff_reg[84] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[84]),
        .Q(async_path_bit[84]),
        .R(1'b0));
  FDRE \src_ff_reg[85] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[85]),
        .Q(async_path_bit[85]),
        .R(1'b0));
  FDRE \src_ff_reg[86] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[86]),
        .Q(async_path_bit[86]),
        .R(1'b0));
  FDRE \src_ff_reg[87] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[87]),
        .Q(async_path_bit[87]),
        .R(1'b0));
  FDRE \src_ff_reg[88] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[88]),
        .Q(async_path_bit[88]),
        .R(1'b0));
  FDRE \src_ff_reg[89] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[89]),
        .Q(async_path_bit[89]),
        .R(1'b0));
  FDRE \src_ff_reg[8] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[8]),
        .Q(async_path_bit[8]),
        .R(1'b0));
  FDRE \src_ff_reg[90] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[90]),
        .Q(async_path_bit[90]),
        .R(1'b0));
  FDRE \src_ff_reg[91] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[91]),
        .Q(async_path_bit[91]),
        .R(1'b0));
  FDRE \src_ff_reg[92] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[92]),
        .Q(async_path_bit[92]),
        .R(1'b0));
  FDRE \src_ff_reg[93] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[93]),
        .Q(async_path_bit[93]),
        .R(1'b0));
  FDRE \src_ff_reg[94] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[94]),
        .Q(async_path_bit[94]),
        .R(1'b0));
  FDRE \src_ff_reg[95] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[95]),
        .Q(async_path_bit[95]),
        .R(1'b0));
  FDRE \src_ff_reg[9] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[9]),
        .Q(async_path_bit[9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[0]),
        .Q(\syncstages_ff[0] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[10]),
        .Q(\syncstages_ff[0] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[11]),
        .Q(\syncstages_ff[0] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[12]),
        .Q(\syncstages_ff[0] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[13]),
        .Q(\syncstages_ff[0] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[14]),
        .Q(\syncstages_ff[0] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[15]),
        .Q(\syncstages_ff[0] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[16]),
        .Q(\syncstages_ff[0] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[17]),
        .Q(\syncstages_ff[0] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[18]),
        .Q(\syncstages_ff[0] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[19]),
        .Q(\syncstages_ff[0] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[1]),
        .Q(\syncstages_ff[0] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[20]),
        .Q(\syncstages_ff[0] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[21]),
        .Q(\syncstages_ff[0] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[22]),
        .Q(\syncstages_ff[0] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[23]),
        .Q(\syncstages_ff[0] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[24]),
        .Q(\syncstages_ff[0] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[25]),
        .Q(\syncstages_ff[0] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[26]),
        .Q(\syncstages_ff[0] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[27]),
        .Q(\syncstages_ff[0] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[28]),
        .Q(\syncstages_ff[0] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[29]),
        .Q(\syncstages_ff[0] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[2]),
        .Q(\syncstages_ff[0] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[30]),
        .Q(\syncstages_ff[0] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[31]),
        .Q(\syncstages_ff[0] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][32] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[32]),
        .Q(\syncstages_ff[0] [32]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][33] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[33]),
        .Q(\syncstages_ff[0] [33]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][34] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[34]),
        .Q(\syncstages_ff[0] [34]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][35] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[35]),
        .Q(\syncstages_ff[0] [35]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][36] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[36]),
        .Q(\syncstages_ff[0] [36]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][37] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[37]),
        .Q(\syncstages_ff[0] [37]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][38] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[38]),
        .Q(\syncstages_ff[0] [38]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][39] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[39]),
        .Q(\syncstages_ff[0] [39]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[3]),
        .Q(\syncstages_ff[0] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][40] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[40]),
        .Q(\syncstages_ff[0] [40]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][41] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[41]),
        .Q(\syncstages_ff[0] [41]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][42] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[42]),
        .Q(\syncstages_ff[0] [42]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][43] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[43]),
        .Q(\syncstages_ff[0] [43]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][44] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[44]),
        .Q(\syncstages_ff[0] [44]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][45] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[45]),
        .Q(\syncstages_ff[0] [45]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][46] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[46]),
        .Q(\syncstages_ff[0] [46]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][47] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[47]),
        .Q(\syncstages_ff[0] [47]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][48] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[48]),
        .Q(\syncstages_ff[0] [48]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][49] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[49]),
        .Q(\syncstages_ff[0] [49]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[4]),
        .Q(\syncstages_ff[0] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][50] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[50]),
        .Q(\syncstages_ff[0] [50]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][51] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[51]),
        .Q(\syncstages_ff[0] [51]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][52] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[52]),
        .Q(\syncstages_ff[0] [52]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][53] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[53]),
        .Q(\syncstages_ff[0] [53]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][54] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[54]),
        .Q(\syncstages_ff[0] [54]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][55] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[55]),
        .Q(\syncstages_ff[0] [55]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][56] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[56]),
        .Q(\syncstages_ff[0] [56]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][57] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[57]),
        .Q(\syncstages_ff[0] [57]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][58] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[58]),
        .Q(\syncstages_ff[0] [58]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][59] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[59]),
        .Q(\syncstages_ff[0] [59]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[5]),
        .Q(\syncstages_ff[0] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][60] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[60]),
        .Q(\syncstages_ff[0] [60]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][61] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[61]),
        .Q(\syncstages_ff[0] [61]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][62] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[62]),
        .Q(\syncstages_ff[0] [62]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][63] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[63]),
        .Q(\syncstages_ff[0] [63]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][64] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[64]),
        .Q(\syncstages_ff[0] [64]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][65] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[65]),
        .Q(\syncstages_ff[0] [65]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][66] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[66]),
        .Q(\syncstages_ff[0] [66]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][67] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[67]),
        .Q(\syncstages_ff[0] [67]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][68] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[68]),
        .Q(\syncstages_ff[0] [68]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][69] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[69]),
        .Q(\syncstages_ff[0] [69]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[6]),
        .Q(\syncstages_ff[0] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][70] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[70]),
        .Q(\syncstages_ff[0] [70]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][71] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[71]),
        .Q(\syncstages_ff[0] [71]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][72] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[72]),
        .Q(\syncstages_ff[0] [72]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][73] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[73]),
        .Q(\syncstages_ff[0] [73]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][74] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[74]),
        .Q(\syncstages_ff[0] [74]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][75] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[75]),
        .Q(\syncstages_ff[0] [75]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][76] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[76]),
        .Q(\syncstages_ff[0] [76]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][77] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[77]),
        .Q(\syncstages_ff[0] [77]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][78] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[78]),
        .Q(\syncstages_ff[0] [78]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][79] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[79]),
        .Q(\syncstages_ff[0] [79]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[7]),
        .Q(\syncstages_ff[0] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][80] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[80]),
        .Q(\syncstages_ff[0] [80]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][81] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[81]),
        .Q(\syncstages_ff[0] [81]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][82] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[82]),
        .Q(\syncstages_ff[0] [82]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][83] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[83]),
        .Q(\syncstages_ff[0] [83]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][84] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[84]),
        .Q(\syncstages_ff[0] [84]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][85] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[85]),
        .Q(\syncstages_ff[0] [85]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][86] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[86]),
        .Q(\syncstages_ff[0] [86]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][87] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[87]),
        .Q(\syncstages_ff[0] [87]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][88] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[88]),
        .Q(\syncstages_ff[0] [88]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][89] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[89]),
        .Q(\syncstages_ff[0] [89]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[8]),
        .Q(\syncstages_ff[0] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][90] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[90]),
        .Q(\syncstages_ff[0] [90]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][91] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[91]),
        .Q(\syncstages_ff[0] [91]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][92] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[92]),
        .Q(\syncstages_ff[0] [92]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][93] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[93]),
        .Q(\syncstages_ff[0] [93]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][94] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[94]),
        .Q(\syncstages_ff[0] [94]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][95] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[95]),
        .Q(\syncstages_ff[0] [95]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[9]),
        .Q(\syncstages_ff[0] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [0]),
        .Q(\syncstages_ff[1] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [10]),
        .Q(\syncstages_ff[1] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [11]),
        .Q(\syncstages_ff[1] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [12]),
        .Q(\syncstages_ff[1] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [13]),
        .Q(\syncstages_ff[1] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [14]),
        .Q(\syncstages_ff[1] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [15]),
        .Q(\syncstages_ff[1] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [16]),
        .Q(\syncstages_ff[1] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [17]),
        .Q(\syncstages_ff[1] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [18]),
        .Q(\syncstages_ff[1] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [19]),
        .Q(\syncstages_ff[1] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [1]),
        .Q(\syncstages_ff[1] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [20]),
        .Q(\syncstages_ff[1] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [21]),
        .Q(\syncstages_ff[1] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [22]),
        .Q(\syncstages_ff[1] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [23]),
        .Q(\syncstages_ff[1] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [24]),
        .Q(\syncstages_ff[1] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [25]),
        .Q(\syncstages_ff[1] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [26]),
        .Q(\syncstages_ff[1] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [27]),
        .Q(\syncstages_ff[1] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [28]),
        .Q(\syncstages_ff[1] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [29]),
        .Q(\syncstages_ff[1] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [2]),
        .Q(\syncstages_ff[1] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [30]),
        .Q(\syncstages_ff[1] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [31]),
        .Q(\syncstages_ff[1] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][32] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [32]),
        .Q(\syncstages_ff[1] [32]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][33] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [33]),
        .Q(\syncstages_ff[1] [33]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][34] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [34]),
        .Q(\syncstages_ff[1] [34]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][35] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [35]),
        .Q(\syncstages_ff[1] [35]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][36] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [36]),
        .Q(\syncstages_ff[1] [36]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][37] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [37]),
        .Q(\syncstages_ff[1] [37]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][38] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [38]),
        .Q(\syncstages_ff[1] [38]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][39] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [39]),
        .Q(\syncstages_ff[1] [39]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [3]),
        .Q(\syncstages_ff[1] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][40] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [40]),
        .Q(\syncstages_ff[1] [40]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][41] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [41]),
        .Q(\syncstages_ff[1] [41]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][42] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [42]),
        .Q(\syncstages_ff[1] [42]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][43] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [43]),
        .Q(\syncstages_ff[1] [43]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][44] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [44]),
        .Q(\syncstages_ff[1] [44]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][45] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [45]),
        .Q(\syncstages_ff[1] [45]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][46] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [46]),
        .Q(\syncstages_ff[1] [46]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][47] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [47]),
        .Q(\syncstages_ff[1] [47]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][48] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [48]),
        .Q(\syncstages_ff[1] [48]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][49] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [49]),
        .Q(\syncstages_ff[1] [49]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [4]),
        .Q(\syncstages_ff[1] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][50] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [50]),
        .Q(\syncstages_ff[1] [50]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][51] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [51]),
        .Q(\syncstages_ff[1] [51]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][52] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [52]),
        .Q(\syncstages_ff[1] [52]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][53] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [53]),
        .Q(\syncstages_ff[1] [53]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][54] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [54]),
        .Q(\syncstages_ff[1] [54]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][55] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [55]),
        .Q(\syncstages_ff[1] [55]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][56] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [56]),
        .Q(\syncstages_ff[1] [56]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][57] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [57]),
        .Q(\syncstages_ff[1] [57]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][58] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [58]),
        .Q(\syncstages_ff[1] [58]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][59] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [59]),
        .Q(\syncstages_ff[1] [59]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [5]),
        .Q(\syncstages_ff[1] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][60] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [60]),
        .Q(\syncstages_ff[1] [60]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][61] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [61]),
        .Q(\syncstages_ff[1] [61]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][62] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [62]),
        .Q(\syncstages_ff[1] [62]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][63] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [63]),
        .Q(\syncstages_ff[1] [63]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][64] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [64]),
        .Q(\syncstages_ff[1] [64]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][65] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [65]),
        .Q(\syncstages_ff[1] [65]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][66] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [66]),
        .Q(\syncstages_ff[1] [66]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][67] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [67]),
        .Q(\syncstages_ff[1] [67]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][68] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [68]),
        .Q(\syncstages_ff[1] [68]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][69] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [69]),
        .Q(\syncstages_ff[1] [69]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [6]),
        .Q(\syncstages_ff[1] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][70] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [70]),
        .Q(\syncstages_ff[1] [70]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][71] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [71]),
        .Q(\syncstages_ff[1] [71]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][72] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [72]),
        .Q(\syncstages_ff[1] [72]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][73] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [73]),
        .Q(\syncstages_ff[1] [73]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][74] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [74]),
        .Q(\syncstages_ff[1] [74]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][75] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [75]),
        .Q(\syncstages_ff[1] [75]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][76] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [76]),
        .Q(\syncstages_ff[1] [76]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][77] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [77]),
        .Q(\syncstages_ff[1] [77]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][78] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [78]),
        .Q(\syncstages_ff[1] [78]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][79] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [79]),
        .Q(\syncstages_ff[1] [79]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [7]),
        .Q(\syncstages_ff[1] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][80] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [80]),
        .Q(\syncstages_ff[1] [80]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][81] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [81]),
        .Q(\syncstages_ff[1] [81]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][82] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [82]),
        .Q(\syncstages_ff[1] [82]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][83] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [83]),
        .Q(\syncstages_ff[1] [83]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][84] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [84]),
        .Q(\syncstages_ff[1] [84]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][85] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [85]),
        .Q(\syncstages_ff[1] [85]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][86] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [86]),
        .Q(\syncstages_ff[1] [86]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][87] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [87]),
        .Q(\syncstages_ff[1] [87]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][88] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [88]),
        .Q(\syncstages_ff[1] [88]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][89] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [89]),
        .Q(\syncstages_ff[1] [89]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [8]),
        .Q(\syncstages_ff[1] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][90] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [90]),
        .Q(\syncstages_ff[1] [90]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][91] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [91]),
        .Q(\syncstages_ff[1] [91]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][92] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [92]),
        .Q(\syncstages_ff[1] [92]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][93] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [93]),
        .Q(\syncstages_ff[1] [93]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][94] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [94]),
        .Q(\syncstages_ff[1] [94]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][95] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [95]),
        .Q(\syncstages_ff[1] [95]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [9]),
        .Q(\syncstages_ff[1] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [0]),
        .Q(\syncstages_ff[2] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [10]),
        .Q(\syncstages_ff[2] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [11]),
        .Q(\syncstages_ff[2] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [12]),
        .Q(\syncstages_ff[2] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [13]),
        .Q(\syncstages_ff[2] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [14]),
        .Q(\syncstages_ff[2] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [15]),
        .Q(\syncstages_ff[2] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [16]),
        .Q(\syncstages_ff[2] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [17]),
        .Q(\syncstages_ff[2] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [18]),
        .Q(\syncstages_ff[2] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [19]),
        .Q(\syncstages_ff[2] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [1]),
        .Q(\syncstages_ff[2] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [20]),
        .Q(\syncstages_ff[2] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [21]),
        .Q(\syncstages_ff[2] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [22]),
        .Q(\syncstages_ff[2] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [23]),
        .Q(\syncstages_ff[2] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [24]),
        .Q(\syncstages_ff[2] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [25]),
        .Q(\syncstages_ff[2] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [26]),
        .Q(\syncstages_ff[2] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [27]),
        .Q(\syncstages_ff[2] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [28]),
        .Q(\syncstages_ff[2] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [29]),
        .Q(\syncstages_ff[2] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [2]),
        .Q(\syncstages_ff[2] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [30]),
        .Q(\syncstages_ff[2] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [31]),
        .Q(\syncstages_ff[2] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][32] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [32]),
        .Q(\syncstages_ff[2] [32]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][33] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [33]),
        .Q(\syncstages_ff[2] [33]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][34] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [34]),
        .Q(\syncstages_ff[2] [34]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][35] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [35]),
        .Q(\syncstages_ff[2] [35]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][36] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [36]),
        .Q(\syncstages_ff[2] [36]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][37] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [37]),
        .Q(\syncstages_ff[2] [37]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][38] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [38]),
        .Q(\syncstages_ff[2] [38]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][39] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [39]),
        .Q(\syncstages_ff[2] [39]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [3]),
        .Q(\syncstages_ff[2] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][40] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [40]),
        .Q(\syncstages_ff[2] [40]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][41] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [41]),
        .Q(\syncstages_ff[2] [41]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][42] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [42]),
        .Q(\syncstages_ff[2] [42]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][43] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [43]),
        .Q(\syncstages_ff[2] [43]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][44] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [44]),
        .Q(\syncstages_ff[2] [44]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][45] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [45]),
        .Q(\syncstages_ff[2] [45]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][46] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [46]),
        .Q(\syncstages_ff[2] [46]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][47] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [47]),
        .Q(\syncstages_ff[2] [47]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][48] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [48]),
        .Q(\syncstages_ff[2] [48]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][49] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [49]),
        .Q(\syncstages_ff[2] [49]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [4]),
        .Q(\syncstages_ff[2] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][50] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [50]),
        .Q(\syncstages_ff[2] [50]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][51] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [51]),
        .Q(\syncstages_ff[2] [51]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][52] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [52]),
        .Q(\syncstages_ff[2] [52]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][53] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [53]),
        .Q(\syncstages_ff[2] [53]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][54] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [54]),
        .Q(\syncstages_ff[2] [54]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][55] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [55]),
        .Q(\syncstages_ff[2] [55]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][56] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [56]),
        .Q(\syncstages_ff[2] [56]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][57] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [57]),
        .Q(\syncstages_ff[2] [57]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][58] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [58]),
        .Q(\syncstages_ff[2] [58]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][59] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [59]),
        .Q(\syncstages_ff[2] [59]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [5]),
        .Q(\syncstages_ff[2] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][60] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [60]),
        .Q(\syncstages_ff[2] [60]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][61] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [61]),
        .Q(\syncstages_ff[2] [61]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][62] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [62]),
        .Q(\syncstages_ff[2] [62]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][63] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [63]),
        .Q(\syncstages_ff[2] [63]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][64] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [64]),
        .Q(\syncstages_ff[2] [64]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][65] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [65]),
        .Q(\syncstages_ff[2] [65]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][66] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [66]),
        .Q(\syncstages_ff[2] [66]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][67] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [67]),
        .Q(\syncstages_ff[2] [67]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][68] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [68]),
        .Q(\syncstages_ff[2] [68]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][69] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [69]),
        .Q(\syncstages_ff[2] [69]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [6]),
        .Q(\syncstages_ff[2] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][70] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [70]),
        .Q(\syncstages_ff[2] [70]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][71] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [71]),
        .Q(\syncstages_ff[2] [71]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][72] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [72]),
        .Q(\syncstages_ff[2] [72]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][73] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [73]),
        .Q(\syncstages_ff[2] [73]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][74] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [74]),
        .Q(\syncstages_ff[2] [74]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][75] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [75]),
        .Q(\syncstages_ff[2] [75]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][76] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [76]),
        .Q(\syncstages_ff[2] [76]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][77] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [77]),
        .Q(\syncstages_ff[2] [77]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][78] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [78]),
        .Q(\syncstages_ff[2] [78]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][79] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [79]),
        .Q(\syncstages_ff[2] [79]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [7]),
        .Q(\syncstages_ff[2] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][80] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [80]),
        .Q(\syncstages_ff[2] [80]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][81] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [81]),
        .Q(\syncstages_ff[2] [81]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][82] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [82]),
        .Q(\syncstages_ff[2] [82]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][83] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [83]),
        .Q(\syncstages_ff[2] [83]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][84] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [84]),
        .Q(\syncstages_ff[2] [84]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][85] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [85]),
        .Q(\syncstages_ff[2] [85]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][86] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [86]),
        .Q(\syncstages_ff[2] [86]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][87] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [87]),
        .Q(\syncstages_ff[2] [87]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][88] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [88]),
        .Q(\syncstages_ff[2] [88]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][89] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [89]),
        .Q(\syncstages_ff[2] [89]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [8]),
        .Q(\syncstages_ff[2] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][90] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [90]),
        .Q(\syncstages_ff[2] [90]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][91] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [91]),
        .Q(\syncstages_ff[2] [91]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][92] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [92]),
        .Q(\syncstages_ff[2] [92]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][93] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [93]),
        .Q(\syncstages_ff[2] [93]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][94] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [94]),
        .Q(\syncstages_ff[2] [94]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][95] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [95]),
        .Q(\syncstages_ff[2] [95]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [9]),
        .Q(\syncstages_ff[2] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [0]),
        .Q(\syncstages_ff[3] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [10]),
        .Q(\syncstages_ff[3] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [11]),
        .Q(\syncstages_ff[3] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [12]),
        .Q(\syncstages_ff[3] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [13]),
        .Q(\syncstages_ff[3] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [14]),
        .Q(\syncstages_ff[3] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [15]),
        .Q(\syncstages_ff[3] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [16]),
        .Q(\syncstages_ff[3] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [17]),
        .Q(\syncstages_ff[3] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [18]),
        .Q(\syncstages_ff[3] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [19]),
        .Q(\syncstages_ff[3] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [1]),
        .Q(\syncstages_ff[3] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [20]),
        .Q(\syncstages_ff[3] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [21]),
        .Q(\syncstages_ff[3] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [22]),
        .Q(\syncstages_ff[3] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [23]),
        .Q(\syncstages_ff[3] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [24]),
        .Q(\syncstages_ff[3] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [25]),
        .Q(\syncstages_ff[3] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [26]),
        .Q(\syncstages_ff[3] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [27]),
        .Q(\syncstages_ff[3] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [28]),
        .Q(\syncstages_ff[3] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [29]),
        .Q(\syncstages_ff[3] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [2]),
        .Q(\syncstages_ff[3] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [30]),
        .Q(\syncstages_ff[3] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [31]),
        .Q(\syncstages_ff[3] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][32] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [32]),
        .Q(\syncstages_ff[3] [32]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][33] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [33]),
        .Q(\syncstages_ff[3] [33]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][34] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [34]),
        .Q(\syncstages_ff[3] [34]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][35] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [35]),
        .Q(\syncstages_ff[3] [35]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][36] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [36]),
        .Q(\syncstages_ff[3] [36]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][37] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [37]),
        .Q(\syncstages_ff[3] [37]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][38] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [38]),
        .Q(\syncstages_ff[3] [38]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][39] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [39]),
        .Q(\syncstages_ff[3] [39]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [3]),
        .Q(\syncstages_ff[3] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][40] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [40]),
        .Q(\syncstages_ff[3] [40]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][41] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [41]),
        .Q(\syncstages_ff[3] [41]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][42] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [42]),
        .Q(\syncstages_ff[3] [42]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][43] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [43]),
        .Q(\syncstages_ff[3] [43]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][44] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [44]),
        .Q(\syncstages_ff[3] [44]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][45] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [45]),
        .Q(\syncstages_ff[3] [45]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][46] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [46]),
        .Q(\syncstages_ff[3] [46]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][47] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [47]),
        .Q(\syncstages_ff[3] [47]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][48] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [48]),
        .Q(\syncstages_ff[3] [48]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][49] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [49]),
        .Q(\syncstages_ff[3] [49]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [4]),
        .Q(\syncstages_ff[3] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][50] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [50]),
        .Q(\syncstages_ff[3] [50]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][51] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [51]),
        .Q(\syncstages_ff[3] [51]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][52] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [52]),
        .Q(\syncstages_ff[3] [52]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][53] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [53]),
        .Q(\syncstages_ff[3] [53]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][54] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [54]),
        .Q(\syncstages_ff[3] [54]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][55] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [55]),
        .Q(\syncstages_ff[3] [55]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][56] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [56]),
        .Q(\syncstages_ff[3] [56]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][57] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [57]),
        .Q(\syncstages_ff[3] [57]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][58] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [58]),
        .Q(\syncstages_ff[3] [58]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][59] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [59]),
        .Q(\syncstages_ff[3] [59]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [5]),
        .Q(\syncstages_ff[3] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][60] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [60]),
        .Q(\syncstages_ff[3] [60]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][61] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [61]),
        .Q(\syncstages_ff[3] [61]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][62] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [62]),
        .Q(\syncstages_ff[3] [62]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][63] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [63]),
        .Q(\syncstages_ff[3] [63]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][64] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [64]),
        .Q(\syncstages_ff[3] [64]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][65] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [65]),
        .Q(\syncstages_ff[3] [65]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][66] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [66]),
        .Q(\syncstages_ff[3] [66]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][67] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [67]),
        .Q(\syncstages_ff[3] [67]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][68] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [68]),
        .Q(\syncstages_ff[3] [68]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][69] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [69]),
        .Q(\syncstages_ff[3] [69]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [6]),
        .Q(\syncstages_ff[3] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][70] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [70]),
        .Q(\syncstages_ff[3] [70]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][71] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [71]),
        .Q(\syncstages_ff[3] [71]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][72] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [72]),
        .Q(\syncstages_ff[3] [72]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][73] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [73]),
        .Q(\syncstages_ff[3] [73]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][74] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [74]),
        .Q(\syncstages_ff[3] [74]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][75] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [75]),
        .Q(\syncstages_ff[3] [75]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][76] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [76]),
        .Q(\syncstages_ff[3] [76]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][77] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [77]),
        .Q(\syncstages_ff[3] [77]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][78] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [78]),
        .Q(\syncstages_ff[3] [78]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][79] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [79]),
        .Q(\syncstages_ff[3] [79]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [7]),
        .Q(\syncstages_ff[3] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][80] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [80]),
        .Q(\syncstages_ff[3] [80]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][81] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [81]),
        .Q(\syncstages_ff[3] [81]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][82] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [82]),
        .Q(\syncstages_ff[3] [82]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][83] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [83]),
        .Q(\syncstages_ff[3] [83]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][84] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [84]),
        .Q(\syncstages_ff[3] [84]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][85] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [85]),
        .Q(\syncstages_ff[3] [85]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][86] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [86]),
        .Q(\syncstages_ff[3] [86]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][87] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [87]),
        .Q(\syncstages_ff[3] [87]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][88] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [88]),
        .Q(\syncstages_ff[3] [88]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][89] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [89]),
        .Q(\syncstages_ff[3] [89]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [8]),
        .Q(\syncstages_ff[3] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][90] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [90]),
        .Q(\syncstages_ff[3] [90]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][91] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [91]),
        .Q(\syncstages_ff[3] [91]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][92] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [92]),
        .Q(\syncstages_ff[3] [92]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][93] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [93]),
        .Q(\syncstages_ff[3] [93]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][94] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [94]),
        .Q(\syncstages_ff[3] [94]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][95] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [95]),
        .Q(\syncstages_ff[3] [95]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [9]),
        .Q(\syncstages_ff[3] [9]),
        .R(1'b0));
endmodule

(* DEST_SYNC_FF = "4" *) (* INIT_SYNC_FF = "0" *) (* ORIG_REF_NAME = "xpm_cdc_array_single" *) 
(* SIM_ASSERT_CHK = "0" *) (* SRC_INPUT_REG = "1" *) (* VERSION = "0" *) 
(* WIDTH = "16" *) (* XPM_MODULE = "TRUE" *) (* xpm_cdc = "ARRAY_SINGLE" *) 
module xpm_cdc_array_single__parameterized3
   (src_clk,
    src_in,
    dest_clk,
    dest_out);
  input src_clk;
  input [15:0]src_in;
  input dest_clk;
  output [15:0]dest_out;

  wire [15:0]async_path_bit;
  wire dest_clk;
  wire src_clk;
  wire [15:0]src_in;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [15:0]\syncstages_ff[0] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [15:0]\syncstages_ff[1] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [15:0]\syncstages_ff[2] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "ARRAY_SINGLE" *) wire [15:0]\syncstages_ff[3] ;

  assign dest_out[15:0] = \syncstages_ff[3] ;
  FDRE \src_ff_reg[0] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[0]),
        .Q(async_path_bit[0]),
        .R(1'b0));
  FDRE \src_ff_reg[10] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[10]),
        .Q(async_path_bit[10]),
        .R(1'b0));
  FDRE \src_ff_reg[11] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[11]),
        .Q(async_path_bit[11]),
        .R(1'b0));
  FDRE \src_ff_reg[12] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[12]),
        .Q(async_path_bit[12]),
        .R(1'b0));
  FDRE \src_ff_reg[13] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[13]),
        .Q(async_path_bit[13]),
        .R(1'b0));
  FDRE \src_ff_reg[14] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[14]),
        .Q(async_path_bit[14]),
        .R(1'b0));
  FDRE \src_ff_reg[15] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[15]),
        .Q(async_path_bit[15]),
        .R(1'b0));
  FDRE \src_ff_reg[1] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[1]),
        .Q(async_path_bit[1]),
        .R(1'b0));
  FDRE \src_ff_reg[2] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[2]),
        .Q(async_path_bit[2]),
        .R(1'b0));
  FDRE \src_ff_reg[3] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[3]),
        .Q(async_path_bit[3]),
        .R(1'b0));
  FDRE \src_ff_reg[4] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[4]),
        .Q(async_path_bit[4]),
        .R(1'b0));
  FDRE \src_ff_reg[5] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[5]),
        .Q(async_path_bit[5]),
        .R(1'b0));
  FDRE \src_ff_reg[6] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[6]),
        .Q(async_path_bit[6]),
        .R(1'b0));
  FDRE \src_ff_reg[7] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[7]),
        .Q(async_path_bit[7]),
        .R(1'b0));
  FDRE \src_ff_reg[8] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[8]),
        .Q(async_path_bit[8]),
        .R(1'b0));
  FDRE \src_ff_reg[9] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in[9]),
        .Q(async_path_bit[9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[0]),
        .Q(\syncstages_ff[0] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[10]),
        .Q(\syncstages_ff[0] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[11]),
        .Q(\syncstages_ff[0] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[12]),
        .Q(\syncstages_ff[0] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[13]),
        .Q(\syncstages_ff[0] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[14]),
        .Q(\syncstages_ff[0] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[15]),
        .Q(\syncstages_ff[0] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[1]),
        .Q(\syncstages_ff[0] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[2]),
        .Q(\syncstages_ff[0] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[3]),
        .Q(\syncstages_ff[0] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[4]),
        .Q(\syncstages_ff[0] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[5]),
        .Q(\syncstages_ff[0] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[6]),
        .Q(\syncstages_ff[0] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[7]),
        .Q(\syncstages_ff[0] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[8]),
        .Q(\syncstages_ff[0] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[0][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path_bit[9]),
        .Q(\syncstages_ff[0] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [0]),
        .Q(\syncstages_ff[1] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [10]),
        .Q(\syncstages_ff[1] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [11]),
        .Q(\syncstages_ff[1] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [12]),
        .Q(\syncstages_ff[1] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [13]),
        .Q(\syncstages_ff[1] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [14]),
        .Q(\syncstages_ff[1] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [15]),
        .Q(\syncstages_ff[1] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [1]),
        .Q(\syncstages_ff[1] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [2]),
        .Q(\syncstages_ff[1] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [3]),
        .Q(\syncstages_ff[1] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [4]),
        .Q(\syncstages_ff[1] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [5]),
        .Q(\syncstages_ff[1] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [6]),
        .Q(\syncstages_ff[1] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [7]),
        .Q(\syncstages_ff[1] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [8]),
        .Q(\syncstages_ff[1] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[1][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[0] [9]),
        .Q(\syncstages_ff[1] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [0]),
        .Q(\syncstages_ff[2] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [10]),
        .Q(\syncstages_ff[2] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [11]),
        .Q(\syncstages_ff[2] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [12]),
        .Q(\syncstages_ff[2] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [13]),
        .Q(\syncstages_ff[2] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [14]),
        .Q(\syncstages_ff[2] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [15]),
        .Q(\syncstages_ff[2] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [1]),
        .Q(\syncstages_ff[2] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [2]),
        .Q(\syncstages_ff[2] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [3]),
        .Q(\syncstages_ff[2] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [4]),
        .Q(\syncstages_ff[2] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [5]),
        .Q(\syncstages_ff[2] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [6]),
        .Q(\syncstages_ff[2] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [7]),
        .Q(\syncstages_ff[2] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [8]),
        .Q(\syncstages_ff[2] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[2][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[1] [9]),
        .Q(\syncstages_ff[2] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [0]),
        .Q(\syncstages_ff[3] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [10]),
        .Q(\syncstages_ff[3] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [11]),
        .Q(\syncstages_ff[3] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [12]),
        .Q(\syncstages_ff[3] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [13]),
        .Q(\syncstages_ff[3] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [14]),
        .Q(\syncstages_ff[3] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [15]),
        .Q(\syncstages_ff[3] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [1]),
        .Q(\syncstages_ff[3] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [2]),
        .Q(\syncstages_ff[3] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [3]),
        .Q(\syncstages_ff[3] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [4]),
        .Q(\syncstages_ff[3] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [5]),
        .Q(\syncstages_ff[3] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [6]),
        .Q(\syncstages_ff[3] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [7]),
        .Q(\syncstages_ff[3] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [8]),
        .Q(\syncstages_ff[3] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "ARRAY_SINGLE" *) 
  FDRE \syncstages_ff_reg[3][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\syncstages_ff[2] [9]),
        .Q(\syncstages_ff[3] [9]),
        .R(1'b0));
endmodule

(* DEST_SYNC_FF = "2" *) (* INIT_SYNC_FF = "0" *) (* REG_OUTPUT = "1" *) 
(* SIM_ASSERT_CHK = "0" *) (* SIM_LOSSLESS_GRAY_CHK = "0" *) (* VERSION = "0" *) 
(* WIDTH = "32" *) (* XPM_MODULE = "TRUE" *) (* xpm_cdc = "GRAY" *) 
module xpm_cdc_gray
   (src_clk,
    src_in_bin,
    dest_clk,
    dest_out_bin);
  input src_clk;
  input [31:0]src_in_bin;
  input dest_clk;
  output [31:0]dest_out_bin;

  wire [31:0]async_path;
  wire [30:0]binval;
  wire dest_clk;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "GRAY" *) wire [31:0]\dest_graysync_ff[0] ;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "GRAY" *) wire [31:0]\dest_graysync_ff[1] ;
  wire [31:0]dest_out_bin;
  wire \dest_out_bin_ff[0]_i_2_n_0 ;
  wire \dest_out_bin_ff[10]_i_2_n_0 ;
  wire \dest_out_bin_ff[14]_i_2_n_0 ;
  wire \dest_out_bin_ff[15]_i_2_n_0 ;
  wire \dest_out_bin_ff[20]_i_2_n_0 ;
  wire \dest_out_bin_ff[25]_i_2_n_0 ;
  wire \dest_out_bin_ff[2]_i_2_n_0 ;
  wire \dest_out_bin_ff[3]_i_2_n_0 ;
  wire \dest_out_bin_ff[4]_i_2_n_0 ;
  wire \dest_out_bin_ff[5]_i_2_n_0 ;
  wire \dest_out_bin_ff[8]_i_2_n_0 ;
  wire \dest_out_bin_ff[9]_i_2_n_0 ;
  wire [30:0]gray_enc;
  wire src_clk;
  wire [31:0]src_in_bin;

  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[0]),
        .Q(\dest_graysync_ff[0] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[10]),
        .Q(\dest_graysync_ff[0] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[11]),
        .Q(\dest_graysync_ff[0] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[12]),
        .Q(\dest_graysync_ff[0] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[13]),
        .Q(\dest_graysync_ff[0] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[14]),
        .Q(\dest_graysync_ff[0] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[15]),
        .Q(\dest_graysync_ff[0] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[16]),
        .Q(\dest_graysync_ff[0] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[17]),
        .Q(\dest_graysync_ff[0] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[18]),
        .Q(\dest_graysync_ff[0] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[19]),
        .Q(\dest_graysync_ff[0] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[1]),
        .Q(\dest_graysync_ff[0] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[20]),
        .Q(\dest_graysync_ff[0] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[21]),
        .Q(\dest_graysync_ff[0] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[22]),
        .Q(\dest_graysync_ff[0] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[23]),
        .Q(\dest_graysync_ff[0] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[24]),
        .Q(\dest_graysync_ff[0] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[25]),
        .Q(\dest_graysync_ff[0] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[26]),
        .Q(\dest_graysync_ff[0] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[27]),
        .Q(\dest_graysync_ff[0] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[28]),
        .Q(\dest_graysync_ff[0] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[29]),
        .Q(\dest_graysync_ff[0] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[2]),
        .Q(\dest_graysync_ff[0] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[30]),
        .Q(\dest_graysync_ff[0] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[31]),
        .Q(\dest_graysync_ff[0] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[3]),
        .Q(\dest_graysync_ff[0] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[4]),
        .Q(\dest_graysync_ff[0] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[5]),
        .Q(\dest_graysync_ff[0] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[6]),
        .Q(\dest_graysync_ff[0] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[7]),
        .Q(\dest_graysync_ff[0] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[8]),
        .Q(\dest_graysync_ff[0] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[0][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(async_path[9]),
        .Q(\dest_graysync_ff[0] [9]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [0]),
        .Q(\dest_graysync_ff[1] [0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [10]),
        .Q(\dest_graysync_ff[1] [10]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [11]),
        .Q(\dest_graysync_ff[1] [11]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [12]),
        .Q(\dest_graysync_ff[1] [12]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [13]),
        .Q(\dest_graysync_ff[1] [13]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [14]),
        .Q(\dest_graysync_ff[1] [14]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [15]),
        .Q(\dest_graysync_ff[1] [15]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [16]),
        .Q(\dest_graysync_ff[1] [16]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [17]),
        .Q(\dest_graysync_ff[1] [17]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [18]),
        .Q(\dest_graysync_ff[1] [18]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [19]),
        .Q(\dest_graysync_ff[1] [19]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [1]),
        .Q(\dest_graysync_ff[1] [1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [20]),
        .Q(\dest_graysync_ff[1] [20]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [21]),
        .Q(\dest_graysync_ff[1] [21]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [22]),
        .Q(\dest_graysync_ff[1] [22]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [23]),
        .Q(\dest_graysync_ff[1] [23]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [24]),
        .Q(\dest_graysync_ff[1] [24]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [25]),
        .Q(\dest_graysync_ff[1] [25]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [26]),
        .Q(\dest_graysync_ff[1] [26]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [27]),
        .Q(\dest_graysync_ff[1] [27]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [28]),
        .Q(\dest_graysync_ff[1] [28]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [29]),
        .Q(\dest_graysync_ff[1] [29]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [2]),
        .Q(\dest_graysync_ff[1] [2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [30]),
        .Q(\dest_graysync_ff[1] [30]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [31]),
        .Q(\dest_graysync_ff[1] [31]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [3]),
        .Q(\dest_graysync_ff[1] [3]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [4]),
        .Q(\dest_graysync_ff[1] [4]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [5]),
        .Q(\dest_graysync_ff[1] [5]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [6]),
        .Q(\dest_graysync_ff[1] [6]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [7]),
        .Q(\dest_graysync_ff[1] [7]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [8]),
        .Q(\dest_graysync_ff[1] [8]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "GRAY" *) 
  FDRE \dest_graysync_ff_reg[1][9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[0] [9]),
        .Q(\dest_graysync_ff[1] [9]),
        .R(1'b0));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[0]_i_1 
       (.I0(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[0]_i_2_n_0 ),
        .I2(\dest_out_bin_ff[8]_i_2_n_0 ),
        .I3(\dest_out_bin_ff[2]_i_2_n_0 ),
        .I4(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I5(\dest_out_bin_ff[14]_i_2_n_0 ),
        .O(binval[0]));
  LUT2 #(
    .INIT(4'h6)) 
    \dest_out_bin_ff[0]_i_2 
       (.I0(\dest_graysync_ff[1] [1]),
        .I1(\dest_graysync_ff[1] [0]),
        .O(\dest_out_bin_ff[0]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[10]_i_1 
       (.I0(\dest_graysync_ff[1] [11]),
        .I1(\dest_out_bin_ff[10]_i_2_n_0 ),
        .I2(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I3(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I4(\dest_graysync_ff[1] [10]),
        .I5(\dest_out_bin_ff[25]_i_2_n_0 ),
        .O(binval[10]));
  LUT2 #(
    .INIT(4'h6)) 
    \dest_out_bin_ff[10]_i_2 
       (.I0(\dest_graysync_ff[1] [13]),
        .I1(\dest_graysync_ff[1] [12]),
        .O(\dest_out_bin_ff[10]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[11]_i_1 
       (.I0(\dest_graysync_ff[1] [12]),
        .I1(\dest_graysync_ff[1] [13]),
        .I2(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I3(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I4(\dest_graysync_ff[1] [11]),
        .I5(\dest_out_bin_ff[25]_i_2_n_0 ),
        .O(binval[11]));
  LUT5 #(
    .INIT(32'h96696996)) 
    \dest_out_bin_ff[12]_i_1 
       (.I0(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I2(\dest_graysync_ff[1] [13]),
        .I3(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I4(\dest_graysync_ff[1] [12]),
        .O(binval[12]));
  LUT4 #(
    .INIT(16'h6996)) 
    \dest_out_bin_ff[13]_i_1 
       (.I0(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I2(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I3(\dest_graysync_ff[1] [13]),
        .O(binval[13]));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT3 #(
    .INIT(8'h96)) 
    \dest_out_bin_ff[14]_i_1 
       (.I0(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I2(\dest_out_bin_ff[25]_i_2_n_0 ),
        .O(binval[14]));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[14]_i_2 
       (.I0(\dest_graysync_ff[1] [15]),
        .I1(\dest_graysync_ff[1] [14]),
        .I2(\dest_graysync_ff[1] [18]),
        .I3(\dest_graysync_ff[1] [19]),
        .I4(\dest_graysync_ff[1] [16]),
        .I5(\dest_graysync_ff[1] [17]),
        .O(\dest_out_bin_ff[14]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[15]_i_1 
       (.I0(\dest_graysync_ff[1] [17]),
        .I1(\dest_out_bin_ff[15]_i_2_n_0 ),
        .I2(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I3(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I4(\dest_graysync_ff[1] [16]),
        .I5(\dest_graysync_ff[1] [15]),
        .O(binval[15]));
  LUT2 #(
    .INIT(4'h6)) 
    \dest_out_bin_ff[15]_i_2 
       (.I0(\dest_graysync_ff[1] [19]),
        .I1(\dest_graysync_ff[1] [18]),
        .O(\dest_out_bin_ff[15]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[16]_i_1 
       (.I0(\dest_graysync_ff[1] [18]),
        .I1(\dest_graysync_ff[1] [19]),
        .I2(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I3(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I4(\dest_graysync_ff[1] [17]),
        .I5(\dest_graysync_ff[1] [16]),
        .O(binval[16]));
  LUT5 #(
    .INIT(32'h96696996)) 
    \dest_out_bin_ff[17]_i_1 
       (.I0(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I2(\dest_graysync_ff[1] [19]),
        .I3(\dest_graysync_ff[1] [17]),
        .I4(\dest_graysync_ff[1] [18]),
        .O(binval[17]));
  LUT4 #(
    .INIT(16'h6996)) 
    \dest_out_bin_ff[18]_i_1 
       (.I0(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I2(\dest_graysync_ff[1] [18]),
        .I3(\dest_graysync_ff[1] [19]),
        .O(binval[18]));
  LUT3 #(
    .INIT(8'h96)) 
    \dest_out_bin_ff[19]_i_1 
       (.I0(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I1(\dest_graysync_ff[1] [19]),
        .I2(\dest_out_bin_ff[25]_i_2_n_0 ),
        .O(binval[19]));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[1]_i_1 
       (.I0(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I1(\dest_graysync_ff[1] [1]),
        .I2(\dest_out_bin_ff[8]_i_2_n_0 ),
        .I3(\dest_out_bin_ff[2]_i_2_n_0 ),
        .I4(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I5(\dest_out_bin_ff[14]_i_2_n_0 ),
        .O(binval[1]));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \dest_out_bin_ff[20]_i_1 
       (.I0(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[20]_i_2_n_0 ),
        .O(binval[20]));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[20]_i_2 
       (.I0(\dest_graysync_ff[1] [21]),
        .I1(\dest_graysync_ff[1] [20]),
        .I2(\dest_graysync_ff[1] [24]),
        .I3(\dest_graysync_ff[1] [25]),
        .I4(\dest_graysync_ff[1] [22]),
        .I5(\dest_graysync_ff[1] [23]),
        .O(\dest_out_bin_ff[20]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[21]_i_1 
       (.I0(\dest_graysync_ff[1] [24]),
        .I1(\dest_graysync_ff[1] [25]),
        .I2(\dest_graysync_ff[1] [21]),
        .I3(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I4(\dest_graysync_ff[1] [23]),
        .I5(\dest_graysync_ff[1] [22]),
        .O(binval[21]));
  LUT5 #(
    .INIT(32'h96696996)) 
    \dest_out_bin_ff[22]_i_1 
       (.I0(\dest_graysync_ff[1] [22]),
        .I1(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I2(\dest_graysync_ff[1] [25]),
        .I3(\dest_graysync_ff[1] [23]),
        .I4(\dest_graysync_ff[1] [24]),
        .O(binval[22]));
  LUT4 #(
    .INIT(16'h6996)) 
    \dest_out_bin_ff[23]_i_1 
       (.I0(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I1(\dest_graysync_ff[1] [23]),
        .I2(\dest_graysync_ff[1] [24]),
        .I3(\dest_graysync_ff[1] [25]),
        .O(binval[23]));
  LUT3 #(
    .INIT(8'h96)) 
    \dest_out_bin_ff[24]_i_1 
       (.I0(\dest_graysync_ff[1] [24]),
        .I1(\dest_graysync_ff[1] [25]),
        .I2(\dest_out_bin_ff[25]_i_2_n_0 ),
        .O(binval[24]));
  LUT2 #(
    .INIT(4'h6)) 
    \dest_out_bin_ff[25]_i_1 
       (.I0(\dest_graysync_ff[1] [25]),
        .I1(\dest_out_bin_ff[25]_i_2_n_0 ),
        .O(binval[25]));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[25]_i_2 
       (.I0(\dest_graysync_ff[1] [27]),
        .I1(\dest_graysync_ff[1] [26]),
        .I2(\dest_graysync_ff[1] [30]),
        .I3(\dest_graysync_ff[1] [31]),
        .I4(\dest_graysync_ff[1] [28]),
        .I5(\dest_graysync_ff[1] [29]),
        .O(\dest_out_bin_ff[25]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[26]_i_1 
       (.I0(\dest_graysync_ff[1] [26]),
        .I1(\dest_graysync_ff[1] [28]),
        .I2(\dest_graysync_ff[1] [30]),
        .I3(\dest_graysync_ff[1] [31]),
        .I4(\dest_graysync_ff[1] [29]),
        .I5(\dest_graysync_ff[1] [27]),
        .O(binval[26]));
  LUT5 #(
    .INIT(32'h96696996)) 
    \dest_out_bin_ff[27]_i_1 
       (.I0(\dest_graysync_ff[1] [27]),
        .I1(\dest_graysync_ff[1] [29]),
        .I2(\dest_graysync_ff[1] [31]),
        .I3(\dest_graysync_ff[1] [30]),
        .I4(\dest_graysync_ff[1] [28]),
        .O(binval[27]));
  LUT4 #(
    .INIT(16'h6996)) 
    \dest_out_bin_ff[28]_i_1 
       (.I0(\dest_graysync_ff[1] [28]),
        .I1(\dest_graysync_ff[1] [30]),
        .I2(\dest_graysync_ff[1] [31]),
        .I3(\dest_graysync_ff[1] [29]),
        .O(binval[28]));
  LUT3 #(
    .INIT(8'h96)) 
    \dest_out_bin_ff[29]_i_1 
       (.I0(\dest_graysync_ff[1] [29]),
        .I1(\dest_graysync_ff[1] [31]),
        .I2(\dest_graysync_ff[1] [30]),
        .O(binval[29]));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT5 #(
    .INIT(32'h96696996)) 
    \dest_out_bin_ff[2]_i_1 
       (.I0(\dest_out_bin_ff[8]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[2]_i_2_n_0 ),
        .I2(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I3(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I4(\dest_out_bin_ff[20]_i_2_n_0 ),
        .O(binval[2]));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[2]_i_2 
       (.I0(\dest_graysync_ff[1] [3]),
        .I1(\dest_graysync_ff[1] [2]),
        .I2(\dest_graysync_ff[1] [6]),
        .I3(\dest_graysync_ff[1] [7]),
        .I4(\dest_graysync_ff[1] [4]),
        .I5(\dest_graysync_ff[1] [5]),
        .O(\dest_out_bin_ff[2]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \dest_out_bin_ff[30]_i_1 
       (.I0(\dest_graysync_ff[1] [30]),
        .I1(\dest_graysync_ff[1] [31]),
        .O(binval[30]));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'h96696996)) 
    \dest_out_bin_ff[3]_i_1 
       (.I0(\dest_out_bin_ff[3]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I2(\dest_out_bin_ff[8]_i_2_n_0 ),
        .I3(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I4(\dest_out_bin_ff[20]_i_2_n_0 ),
        .O(binval[3]));
  LUT5 #(
    .INIT(32'h96696996)) 
    \dest_out_bin_ff[3]_i_2 
       (.I0(\dest_graysync_ff[1] [7]),
        .I1(\dest_graysync_ff[1] [6]),
        .I2(\dest_graysync_ff[1] [5]),
        .I3(\dest_graysync_ff[1] [4]),
        .I4(\dest_graysync_ff[1] [3]),
        .O(\dest_out_bin_ff[3]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'h96696996)) 
    \dest_out_bin_ff[4]_i_1 
       (.I0(\dest_out_bin_ff[4]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I2(\dest_out_bin_ff[8]_i_2_n_0 ),
        .I3(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I4(\dest_out_bin_ff[20]_i_2_n_0 ),
        .O(binval[4]));
  LUT4 #(
    .INIT(16'h6996)) 
    \dest_out_bin_ff[4]_i_2 
       (.I0(\dest_graysync_ff[1] [5]),
        .I1(\dest_graysync_ff[1] [7]),
        .I2(\dest_graysync_ff[1] [6]),
        .I3(\dest_graysync_ff[1] [4]),
        .O(\dest_out_bin_ff[4]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[5]_i_1 
       (.I0(\dest_graysync_ff[1] [5]),
        .I1(\dest_out_bin_ff[5]_i_2_n_0 ),
        .I2(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I3(\dest_out_bin_ff[8]_i_2_n_0 ),
        .I4(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I5(\dest_out_bin_ff[20]_i_2_n_0 ),
        .O(binval[5]));
  LUT2 #(
    .INIT(4'h6)) 
    \dest_out_bin_ff[5]_i_2 
       (.I0(\dest_graysync_ff[1] [7]),
        .I1(\dest_graysync_ff[1] [6]),
        .O(\dest_out_bin_ff[5]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[6]_i_1 
       (.I0(\dest_graysync_ff[1] [6]),
        .I1(\dest_graysync_ff[1] [7]),
        .I2(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I3(\dest_out_bin_ff[8]_i_2_n_0 ),
        .I4(\dest_out_bin_ff[25]_i_2_n_0 ),
        .I5(\dest_out_bin_ff[20]_i_2_n_0 ),
        .O(binval[6]));
  LUT5 #(
    .INIT(32'h96696996)) 
    \dest_out_bin_ff[7]_i_1 
       (.I0(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[8]_i_2_n_0 ),
        .I2(\dest_graysync_ff[1] [7]),
        .I3(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I4(\dest_out_bin_ff[25]_i_2_n_0 ),
        .O(binval[7]));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT4 #(
    .INIT(16'h6996)) 
    \dest_out_bin_ff[8]_i_1 
       (.I0(\dest_out_bin_ff[8]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I2(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I3(\dest_out_bin_ff[25]_i_2_n_0 ),
        .O(binval[8]));
  LUT6 #(
    .INIT(64'h6996966996696996)) 
    \dest_out_bin_ff[8]_i_2 
       (.I0(\dest_graysync_ff[1] [9]),
        .I1(\dest_graysync_ff[1] [8]),
        .I2(\dest_graysync_ff[1] [12]),
        .I3(\dest_graysync_ff[1] [13]),
        .I4(\dest_graysync_ff[1] [10]),
        .I5(\dest_graysync_ff[1] [11]),
        .O(\dest_out_bin_ff[8]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'h96696996)) 
    \dest_out_bin_ff[9]_i_1 
       (.I0(\dest_out_bin_ff[9]_i_2_n_0 ),
        .I1(\dest_out_bin_ff[20]_i_2_n_0 ),
        .I2(\dest_out_bin_ff[14]_i_2_n_0 ),
        .I3(\dest_graysync_ff[1] [9]),
        .I4(\dest_out_bin_ff[25]_i_2_n_0 ),
        .O(binval[9]));
  LUT4 #(
    .INIT(16'h6996)) 
    \dest_out_bin_ff[9]_i_2 
       (.I0(\dest_graysync_ff[1] [11]),
        .I1(\dest_graysync_ff[1] [13]),
        .I2(\dest_graysync_ff[1] [12]),
        .I3(\dest_graysync_ff[1] [10]),
        .O(\dest_out_bin_ff[9]_i_2_n_0 ));
  FDRE \dest_out_bin_ff_reg[0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[0]),
        .Q(dest_out_bin[0]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[10] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[10]),
        .Q(dest_out_bin[10]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[11] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[11]),
        .Q(dest_out_bin[11]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[12] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[12]),
        .Q(dest_out_bin[12]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[13] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[13]),
        .Q(dest_out_bin[13]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[14] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[14]),
        .Q(dest_out_bin[14]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[15] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[15]),
        .Q(dest_out_bin[15]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[16] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[16]),
        .Q(dest_out_bin[16]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[17] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[17]),
        .Q(dest_out_bin[17]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[18] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[18]),
        .Q(dest_out_bin[18]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[19] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[19]),
        .Q(dest_out_bin[19]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[1]),
        .Q(dest_out_bin[1]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[20] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[20]),
        .Q(dest_out_bin[20]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[21] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[21]),
        .Q(dest_out_bin[21]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[22] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[22]),
        .Q(dest_out_bin[22]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[23] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[23]),
        .Q(dest_out_bin[23]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[24] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[24]),
        .Q(dest_out_bin[24]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[25] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[25]),
        .Q(dest_out_bin[25]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[26] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[26]),
        .Q(dest_out_bin[26]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[27] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[27]),
        .Q(dest_out_bin[27]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[28] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[28]),
        .Q(dest_out_bin[28]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[29] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[29]),
        .Q(dest_out_bin[29]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[2]),
        .Q(dest_out_bin[2]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[30] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[30]),
        .Q(dest_out_bin[30]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[31] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(\dest_graysync_ff[1] [31]),
        .Q(dest_out_bin[31]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[3]),
        .Q(dest_out_bin[3]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[4] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[4]),
        .Q(dest_out_bin[4]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[5] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[5]),
        .Q(dest_out_bin[5]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[6] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[6]),
        .Q(dest_out_bin[6]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[7] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[7]),
        .Q(dest_out_bin[7]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[8] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[8]),
        .Q(dest_out_bin[8]),
        .R(1'b0));
  FDRE \dest_out_bin_ff_reg[9] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(binval[9]),
        .Q(dest_out_bin[9]),
        .R(1'b0));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[0]_i_1 
       (.I0(src_in_bin[1]),
        .I1(src_in_bin[0]),
        .O(gray_enc[0]));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[10]_i_1 
       (.I0(src_in_bin[11]),
        .I1(src_in_bin[10]),
        .O(gray_enc[10]));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[11]_i_1 
       (.I0(src_in_bin[12]),
        .I1(src_in_bin[11]),
        .O(gray_enc[11]));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[12]_i_1 
       (.I0(src_in_bin[13]),
        .I1(src_in_bin[12]),
        .O(gray_enc[12]));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[13]_i_1 
       (.I0(src_in_bin[14]),
        .I1(src_in_bin[13]),
        .O(gray_enc[13]));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[14]_i_1 
       (.I0(src_in_bin[15]),
        .I1(src_in_bin[14]),
        .O(gray_enc[14]));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[15]_i_1 
       (.I0(src_in_bin[16]),
        .I1(src_in_bin[15]),
        .O(gray_enc[15]));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[16]_i_1 
       (.I0(src_in_bin[17]),
        .I1(src_in_bin[16]),
        .O(gray_enc[16]));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[17]_i_1 
       (.I0(src_in_bin[18]),
        .I1(src_in_bin[17]),
        .O(gray_enc[17]));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[18]_i_1 
       (.I0(src_in_bin[19]),
        .I1(src_in_bin[18]),
        .O(gray_enc[18]));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[19]_i_1 
       (.I0(src_in_bin[20]),
        .I1(src_in_bin[19]),
        .O(gray_enc[19]));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[1]_i_1 
       (.I0(src_in_bin[2]),
        .I1(src_in_bin[1]),
        .O(gray_enc[1]));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[20]_i_1 
       (.I0(src_in_bin[21]),
        .I1(src_in_bin[20]),
        .O(gray_enc[20]));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[21]_i_1 
       (.I0(src_in_bin[22]),
        .I1(src_in_bin[21]),
        .O(gray_enc[21]));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[22]_i_1 
       (.I0(src_in_bin[23]),
        .I1(src_in_bin[22]),
        .O(gray_enc[22]));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[23]_i_1 
       (.I0(src_in_bin[24]),
        .I1(src_in_bin[23]),
        .O(gray_enc[23]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[24]_i_1 
       (.I0(src_in_bin[25]),
        .I1(src_in_bin[24]),
        .O(gray_enc[24]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[25]_i_1 
       (.I0(src_in_bin[26]),
        .I1(src_in_bin[25]),
        .O(gray_enc[25]));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[26]_i_1 
       (.I0(src_in_bin[27]),
        .I1(src_in_bin[26]),
        .O(gray_enc[26]));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[27]_i_1 
       (.I0(src_in_bin[28]),
        .I1(src_in_bin[27]),
        .O(gray_enc[27]));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[28]_i_1 
       (.I0(src_in_bin[29]),
        .I1(src_in_bin[28]),
        .O(gray_enc[28]));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[29]_i_1 
       (.I0(src_in_bin[30]),
        .I1(src_in_bin[29]),
        .O(gray_enc[29]));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[2]_i_1 
       (.I0(src_in_bin[3]),
        .I1(src_in_bin[2]),
        .O(gray_enc[2]));
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[30]_i_1 
       (.I0(src_in_bin[31]),
        .I1(src_in_bin[30]),
        .O(gray_enc[30]));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[3]_i_1 
       (.I0(src_in_bin[4]),
        .I1(src_in_bin[3]),
        .O(gray_enc[3]));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[4]_i_1 
       (.I0(src_in_bin[5]),
        .I1(src_in_bin[4]),
        .O(gray_enc[4]));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[5]_i_1 
       (.I0(src_in_bin[6]),
        .I1(src_in_bin[5]),
        .O(gray_enc[5]));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[6]_i_1 
       (.I0(src_in_bin[7]),
        .I1(src_in_bin[6]),
        .O(gray_enc[6]));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[7]_i_1 
       (.I0(src_in_bin[8]),
        .I1(src_in_bin[7]),
        .O(gray_enc[7]));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[8]_i_1 
       (.I0(src_in_bin[9]),
        .I1(src_in_bin[8]),
        .O(gray_enc[8]));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \src_gray_ff[9]_i_1 
       (.I0(src_in_bin[10]),
        .I1(src_in_bin[9]),
        .O(gray_enc[9]));
  FDRE \src_gray_ff_reg[0] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[0]),
        .Q(async_path[0]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[10] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[10]),
        .Q(async_path[10]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[11] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[11]),
        .Q(async_path[11]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[12] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[12]),
        .Q(async_path[12]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[13] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[13]),
        .Q(async_path[13]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[14] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[14]),
        .Q(async_path[14]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[15] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[15]),
        .Q(async_path[15]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[16] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[16]),
        .Q(async_path[16]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[17] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[17]),
        .Q(async_path[17]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[18] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[18]),
        .Q(async_path[18]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[19] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[19]),
        .Q(async_path[19]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[1] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[1]),
        .Q(async_path[1]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[20] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[20]),
        .Q(async_path[20]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[21] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[21]),
        .Q(async_path[21]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[22] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[22]),
        .Q(async_path[22]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[23] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[23]),
        .Q(async_path[23]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[24] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[24]),
        .Q(async_path[24]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[25] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[25]),
        .Q(async_path[25]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[26] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[26]),
        .Q(async_path[26]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[27] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[27]),
        .Q(async_path[27]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[28] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[28]),
        .Q(async_path[28]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[29] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[29]),
        .Q(async_path[29]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[2] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[2]),
        .Q(async_path[2]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[30] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[30]),
        .Q(async_path[30]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[31] 
       (.C(src_clk),
        .CE(1'b1),
        .D(src_in_bin[31]),
        .Q(async_path[31]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[3] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[3]),
        .Q(async_path[3]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[4] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[4]),
        .Q(async_path[4]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[5] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[5]),
        .Q(async_path[5]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[6] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[6]),
        .Q(async_path[6]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[7] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[7]),
        .Q(async_path[7]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[8] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[8]),
        .Q(async_path[8]),
        .R(1'b0));
  FDRE \src_gray_ff_reg[9] 
       (.C(src_clk),
        .CE(1'b1),
        .D(gray_enc[9]),
        .Q(async_path[9]),
        .R(1'b0));
endmodule

(* DEF_VAL = "1'b1" *) (* DEST_SYNC_FF = "4" *) (* INIT = "1" *) 
(* INIT_SYNC_FF = "0" *) (* SIM_ASSERT_CHK = "0" *) (* VERSION = "0" *) 
(* XPM_MODULE = "TRUE" *) (* xpm_cdc = "SYNC_RST" *) 
module xpm_cdc_sync_rst
   (src_rst,
    dest_clk,
    dest_rst);
  input src_rst;
  input dest_clk;
  output dest_rst;

  wire dest_clk;
  wire src_rst;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "SYNC_RST" *) wire [3:0]syncstages_ff;

  assign dest_rst = syncstages_ff[3];
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SYNC_RST" *) 
  FDRE #(
    .INIT(1'b1)) 
    \syncstages_ff_reg[0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(src_rst),
        .Q(syncstages_ff[0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SYNC_RST" *) 
  FDRE #(
    .INIT(1'b1)) 
    \syncstages_ff_reg[1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(syncstages_ff[0]),
        .Q(syncstages_ff[1]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SYNC_RST" *) 
  FDRE #(
    .INIT(1'b1)) 
    \syncstages_ff_reg[2] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(syncstages_ff[1]),
        .Q(syncstages_ff[2]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SYNC_RST" *) 
  FDRE #(
    .INIT(1'b1)) 
    \syncstages_ff_reg[3] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(syncstages_ff[2]),
        .Q(syncstages_ff[3]),
        .R(1'b0));
endmodule
`ifndef GLBL
`define GLBL
`timescale  1 ps / 1 ps

module glbl ();

    parameter ROC_WIDTH = 100000;
    parameter TOC_WIDTH = 0;

//--------   STARTUP Globals --------------
    wire GSR;
    wire GTS;
    wire GWE;
    wire PRLD;
    tri1 p_up_tmp;
    tri (weak1, strong0) PLL_LOCKG = p_up_tmp;

    wire PROGB_GLBL;
    wire CCLKO_GLBL;
    wire FCSBO_GLBL;
    wire [3:0] DO_GLBL;
    wire [3:0] DI_GLBL;
   
    reg GSR_int;
    reg GTS_int;
    reg PRLD_int;

//--------   JTAG Globals --------------
    wire JTAG_TDO_GLBL;
    wire JTAG_TCK_GLBL;
    wire JTAG_TDI_GLBL;
    wire JTAG_TMS_GLBL;
    wire JTAG_TRST_GLBL;

    reg JTAG_CAPTURE_GLBL;
    reg JTAG_RESET_GLBL;
    reg JTAG_SHIFT_GLBL;
    reg JTAG_UPDATE_GLBL;
    reg JTAG_RUNTEST_GLBL;

    reg JTAG_SEL1_GLBL = 0;
    reg JTAG_SEL2_GLBL = 0 ;
    reg JTAG_SEL3_GLBL = 0;
    reg JTAG_SEL4_GLBL = 0;

    reg JTAG_USER_TDO1_GLBL = 1'bz;
    reg JTAG_USER_TDO2_GLBL = 1'bz;
    reg JTAG_USER_TDO3_GLBL = 1'bz;
    reg JTAG_USER_TDO4_GLBL = 1'bz;

    assign (strong1, weak0) GSR = GSR_int;
    assign (strong1, weak0) GTS = GTS_int;
    assign (weak1, weak0) PRLD = PRLD_int;

    initial begin
	GSR_int = 1'b1;
	PRLD_int = 1'b1;
	#(ROC_WIDTH)
	GSR_int = 1'b0;
	PRLD_int = 1'b0;
    end

    initial begin
	GTS_int = 1'b1;
	#(TOC_WIDTH)
	GTS_int = 1'b0;
    end

endmodule
`endif
