// Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2018.3 (lin64) Build 2405991 Thu Dec  6 23:36:41 MST 2018
// Date        : Wed Nov 18 11:00:00 2020
// Host        : bender.ad.sklk.us running 64-bit Ubuntu 16.04.6 LTS
// Command     : write_verilog -force -mode funcsim time_master_sim.v
// Design      : time_master
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7z030sbg485-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* NotValidForBitStream *)
module time_master
   (SYS_clk,
    SYS_rst,
    SYS_time_in,
    SYS_time_out,
    SYS_time_read,
    SYS_time_write,
    SYS_time_asap,
    DATA_clk,
    DATA_rst,
    DATA_stb,
    DATA_trigger_in,
    DATA_time_now);
  input SYS_clk;
  input SYS_rst;
  input [63:0]SYS_time_in;
  output [63:0]SYS_time_out;
  input SYS_time_read;
  input SYS_time_write;
  input SYS_time_asap;
  input DATA_clk;
  input DATA_rst;
  input DATA_stb;
  input DATA_trigger_in;
  output [63:0]DATA_time_now;

  wire DATA_clk;
  wire DATA_rst;
  wire DATA_stb;
  wire DATA_time_asap;
  wire [63:0]DATA_time_last;
  wire [63:0]DATA_time_next;
  wire [63:0]DATA_time_now;
  wire DATA_trigger_in;
  wire SYS_clk;
  wire SYS_rst;
  wire SYS_time_asap;
  wire [63:0]SYS_time_in;
  wire [63:0]SYS_time_out;
  wire \SYS_time_out[0]_i_1_n_0 ;
  wire \SYS_time_out[10]_i_1_n_0 ;
  wire \SYS_time_out[11]_i_1_n_0 ;
  wire \SYS_time_out[12]_i_1_n_0 ;
  wire \SYS_time_out[13]_i_1_n_0 ;
  wire \SYS_time_out[14]_i_1_n_0 ;
  wire \SYS_time_out[15]_i_1_n_0 ;
  wire \SYS_time_out[16]_i_1_n_0 ;
  wire \SYS_time_out[17]_i_1_n_0 ;
  wire \SYS_time_out[18]_i_1_n_0 ;
  wire \SYS_time_out[19]_i_1_n_0 ;
  wire \SYS_time_out[1]_i_1_n_0 ;
  wire \SYS_time_out[20]_i_1_n_0 ;
  wire \SYS_time_out[21]_i_1_n_0 ;
  wire \SYS_time_out[22]_i_1_n_0 ;
  wire \SYS_time_out[23]_i_1_n_0 ;
  wire \SYS_time_out[24]_i_1_n_0 ;
  wire \SYS_time_out[25]_i_1_n_0 ;
  wire \SYS_time_out[26]_i_1_n_0 ;
  wire \SYS_time_out[27]_i_1_n_0 ;
  wire \SYS_time_out[28]_i_1_n_0 ;
  wire \SYS_time_out[29]_i_1_n_0 ;
  wire \SYS_time_out[2]_i_1_n_0 ;
  wire \SYS_time_out[30]_i_1_n_0 ;
  wire \SYS_time_out[31]_i_1_n_0 ;
  wire \SYS_time_out[32]_i_1_n_0 ;
  wire \SYS_time_out[33]_i_1_n_0 ;
  wire \SYS_time_out[34]_i_1_n_0 ;
  wire \SYS_time_out[35]_i_1_n_0 ;
  wire \SYS_time_out[36]_i_1_n_0 ;
  wire \SYS_time_out[37]_i_1_n_0 ;
  wire \SYS_time_out[38]_i_1_n_0 ;
  wire \SYS_time_out[39]_i_1_n_0 ;
  wire \SYS_time_out[3]_i_1_n_0 ;
  wire \SYS_time_out[40]_i_1_n_0 ;
  wire \SYS_time_out[41]_i_1_n_0 ;
  wire \SYS_time_out[42]_i_1_n_0 ;
  wire \SYS_time_out[43]_i_1_n_0 ;
  wire \SYS_time_out[44]_i_1_n_0 ;
  wire \SYS_time_out[45]_i_1_n_0 ;
  wire \SYS_time_out[46]_i_1_n_0 ;
  wire \SYS_time_out[47]_i_1_n_0 ;
  wire \SYS_time_out[48]_i_1_n_0 ;
  wire \SYS_time_out[49]_i_1_n_0 ;
  wire \SYS_time_out[4]_i_1_n_0 ;
  wire \SYS_time_out[50]_i_1_n_0 ;
  wire \SYS_time_out[51]_i_1_n_0 ;
  wire \SYS_time_out[52]_i_1_n_0 ;
  wire \SYS_time_out[53]_i_1_n_0 ;
  wire \SYS_time_out[54]_i_1_n_0 ;
  wire \SYS_time_out[55]_i_1_n_0 ;
  wire \SYS_time_out[56]_i_1_n_0 ;
  wire \SYS_time_out[57]_i_1_n_0 ;
  wire \SYS_time_out[58]_i_1_n_0 ;
  wire \SYS_time_out[59]_i_1_n_0 ;
  wire \SYS_time_out[5]_i_1_n_0 ;
  wire \SYS_time_out[60]_i_1_n_0 ;
  wire \SYS_time_out[61]_i_1_n_0 ;
  wire \SYS_time_out[62]_i_1_n_0 ;
  wire \SYS_time_out[63]_i_1_n_0 ;
  wire \SYS_time_out[6]_i_1_n_0 ;
  wire \SYS_time_out[7]_i_1_n_0 ;
  wire \SYS_time_out[8]_i_1_n_0 ;
  wire \SYS_time_out[9]_i_1_n_0 ;
  wire SYS_time_read;
  wire SYS_time_write;
  wire dest_ack_i_1__0_n_0;
  wire dest_ack_i_1__1_n_0;
  wire dest_ack_i_1_n_0;
  wire has_next_time;
  wire has_next_time_i_1_n_0;
  wire [63:0]next_time;
  wire next_time0;
  wire next_time_asap_i_1_n_0;
  wire next_time_asap_reg_n_0;
  wire [63:0]out_time;
  wire \src_in[63]_i_1__0_n_0 ;
  wire \src_in[63]_i_1_n_0 ;
  wire src_send_i_1__0_n_0;
  wire src_send_i_1__1_n_0;
  wire src_send_i_1_n_0;
  wire \time_counter[11]_i_2_n_0 ;
  wire \time_counter[11]_i_3_n_0 ;
  wire \time_counter[11]_i_4_n_0 ;
  wire \time_counter[11]_i_5_n_0 ;
  wire \time_counter[15]_i_2_n_0 ;
  wire \time_counter[15]_i_3_n_0 ;
  wire \time_counter[15]_i_4_n_0 ;
  wire \time_counter[15]_i_5_n_0 ;
  wire \time_counter[19]_i_2_n_0 ;
  wire \time_counter[19]_i_3_n_0 ;
  wire \time_counter[19]_i_4_n_0 ;
  wire \time_counter[19]_i_5_n_0 ;
  wire \time_counter[23]_i_2_n_0 ;
  wire \time_counter[23]_i_3_n_0 ;
  wire \time_counter[23]_i_4_n_0 ;
  wire \time_counter[23]_i_5_n_0 ;
  wire \time_counter[27]_i_2_n_0 ;
  wire \time_counter[27]_i_3_n_0 ;
  wire \time_counter[27]_i_4_n_0 ;
  wire \time_counter[27]_i_5_n_0 ;
  wire \time_counter[31]_i_2_n_0 ;
  wire \time_counter[31]_i_3_n_0 ;
  wire \time_counter[31]_i_4_n_0 ;
  wire \time_counter[31]_i_5_n_0 ;
  wire \time_counter[35]_i_2_n_0 ;
  wire \time_counter[35]_i_3_n_0 ;
  wire \time_counter[35]_i_4_n_0 ;
  wire \time_counter[35]_i_5_n_0 ;
  wire \time_counter[39]_i_2_n_0 ;
  wire \time_counter[39]_i_3_n_0 ;
  wire \time_counter[39]_i_4_n_0 ;
  wire \time_counter[39]_i_5_n_0 ;
  wire \time_counter[3]_i_2_n_0 ;
  wire \time_counter[3]_i_3_n_0 ;
  wire \time_counter[3]_i_4_n_0 ;
  wire \time_counter[3]_i_5_n_0 ;
  wire \time_counter[3]_i_6_n_0 ;
  wire \time_counter[43]_i_2_n_0 ;
  wire \time_counter[43]_i_3_n_0 ;
  wire \time_counter[43]_i_4_n_0 ;
  wire \time_counter[43]_i_5_n_0 ;
  wire \time_counter[47]_i_2_n_0 ;
  wire \time_counter[47]_i_3_n_0 ;
  wire \time_counter[47]_i_4_n_0 ;
  wire \time_counter[47]_i_5_n_0 ;
  wire \time_counter[51]_i_2_n_0 ;
  wire \time_counter[51]_i_3_n_0 ;
  wire \time_counter[51]_i_4_n_0 ;
  wire \time_counter[51]_i_5_n_0 ;
  wire \time_counter[55]_i_2_n_0 ;
  wire \time_counter[55]_i_3_n_0 ;
  wire \time_counter[55]_i_4_n_0 ;
  wire \time_counter[55]_i_5_n_0 ;
  wire \time_counter[59]_i_2_n_0 ;
  wire \time_counter[59]_i_3_n_0 ;
  wire \time_counter[59]_i_4_n_0 ;
  wire \time_counter[59]_i_5_n_0 ;
  wire \time_counter[63]_i_2_n_0 ;
  wire \time_counter[63]_i_3_n_0 ;
  wire \time_counter[63]_i_4_n_0 ;
  wire \time_counter[63]_i_5_n_0 ;
  wire \time_counter[7]_i_2_n_0 ;
  wire \time_counter[7]_i_3_n_0 ;
  wire \time_counter[7]_i_4_n_0 ;
  wire \time_counter[7]_i_5_n_0 ;
  wire \time_counter_reg[11]_i_1_n_0 ;
  wire \time_counter_reg[11]_i_1_n_1 ;
  wire \time_counter_reg[11]_i_1_n_2 ;
  wire \time_counter_reg[11]_i_1_n_3 ;
  wire \time_counter_reg[11]_i_1_n_4 ;
  wire \time_counter_reg[11]_i_1_n_5 ;
  wire \time_counter_reg[11]_i_1_n_6 ;
  wire \time_counter_reg[11]_i_1_n_7 ;
  wire \time_counter_reg[15]_i_1_n_0 ;
  wire \time_counter_reg[15]_i_1_n_1 ;
  wire \time_counter_reg[15]_i_1_n_2 ;
  wire \time_counter_reg[15]_i_1_n_3 ;
  wire \time_counter_reg[15]_i_1_n_4 ;
  wire \time_counter_reg[15]_i_1_n_5 ;
  wire \time_counter_reg[15]_i_1_n_6 ;
  wire \time_counter_reg[15]_i_1_n_7 ;
  wire \time_counter_reg[19]_i_1_n_0 ;
  wire \time_counter_reg[19]_i_1_n_1 ;
  wire \time_counter_reg[19]_i_1_n_2 ;
  wire \time_counter_reg[19]_i_1_n_3 ;
  wire \time_counter_reg[19]_i_1_n_4 ;
  wire \time_counter_reg[19]_i_1_n_5 ;
  wire \time_counter_reg[19]_i_1_n_6 ;
  wire \time_counter_reg[19]_i_1_n_7 ;
  wire \time_counter_reg[23]_i_1_n_0 ;
  wire \time_counter_reg[23]_i_1_n_1 ;
  wire \time_counter_reg[23]_i_1_n_2 ;
  wire \time_counter_reg[23]_i_1_n_3 ;
  wire \time_counter_reg[23]_i_1_n_4 ;
  wire \time_counter_reg[23]_i_1_n_5 ;
  wire \time_counter_reg[23]_i_1_n_6 ;
  wire \time_counter_reg[23]_i_1_n_7 ;
  wire \time_counter_reg[27]_i_1_n_0 ;
  wire \time_counter_reg[27]_i_1_n_1 ;
  wire \time_counter_reg[27]_i_1_n_2 ;
  wire \time_counter_reg[27]_i_1_n_3 ;
  wire \time_counter_reg[27]_i_1_n_4 ;
  wire \time_counter_reg[27]_i_1_n_5 ;
  wire \time_counter_reg[27]_i_1_n_6 ;
  wire \time_counter_reg[27]_i_1_n_7 ;
  wire \time_counter_reg[31]_i_1_n_0 ;
  wire \time_counter_reg[31]_i_1_n_1 ;
  wire \time_counter_reg[31]_i_1_n_2 ;
  wire \time_counter_reg[31]_i_1_n_3 ;
  wire \time_counter_reg[31]_i_1_n_4 ;
  wire \time_counter_reg[31]_i_1_n_5 ;
  wire \time_counter_reg[31]_i_1_n_6 ;
  wire \time_counter_reg[31]_i_1_n_7 ;
  wire \time_counter_reg[35]_i_1_n_0 ;
  wire \time_counter_reg[35]_i_1_n_1 ;
  wire \time_counter_reg[35]_i_1_n_2 ;
  wire \time_counter_reg[35]_i_1_n_3 ;
  wire \time_counter_reg[35]_i_1_n_4 ;
  wire \time_counter_reg[35]_i_1_n_5 ;
  wire \time_counter_reg[35]_i_1_n_6 ;
  wire \time_counter_reg[35]_i_1_n_7 ;
  wire \time_counter_reg[39]_i_1_n_0 ;
  wire \time_counter_reg[39]_i_1_n_1 ;
  wire \time_counter_reg[39]_i_1_n_2 ;
  wire \time_counter_reg[39]_i_1_n_3 ;
  wire \time_counter_reg[39]_i_1_n_4 ;
  wire \time_counter_reg[39]_i_1_n_5 ;
  wire \time_counter_reg[39]_i_1_n_6 ;
  wire \time_counter_reg[39]_i_1_n_7 ;
  wire \time_counter_reg[3]_i_1_n_0 ;
  wire \time_counter_reg[3]_i_1_n_1 ;
  wire \time_counter_reg[3]_i_1_n_2 ;
  wire \time_counter_reg[3]_i_1_n_3 ;
  wire \time_counter_reg[3]_i_1_n_4 ;
  wire \time_counter_reg[3]_i_1_n_5 ;
  wire \time_counter_reg[3]_i_1_n_6 ;
  wire \time_counter_reg[3]_i_1_n_7 ;
  wire \time_counter_reg[43]_i_1_n_0 ;
  wire \time_counter_reg[43]_i_1_n_1 ;
  wire \time_counter_reg[43]_i_1_n_2 ;
  wire \time_counter_reg[43]_i_1_n_3 ;
  wire \time_counter_reg[43]_i_1_n_4 ;
  wire \time_counter_reg[43]_i_1_n_5 ;
  wire \time_counter_reg[43]_i_1_n_6 ;
  wire \time_counter_reg[43]_i_1_n_7 ;
  wire \time_counter_reg[47]_i_1_n_0 ;
  wire \time_counter_reg[47]_i_1_n_1 ;
  wire \time_counter_reg[47]_i_1_n_2 ;
  wire \time_counter_reg[47]_i_1_n_3 ;
  wire \time_counter_reg[47]_i_1_n_4 ;
  wire \time_counter_reg[47]_i_1_n_5 ;
  wire \time_counter_reg[47]_i_1_n_6 ;
  wire \time_counter_reg[47]_i_1_n_7 ;
  wire \time_counter_reg[51]_i_1_n_0 ;
  wire \time_counter_reg[51]_i_1_n_1 ;
  wire \time_counter_reg[51]_i_1_n_2 ;
  wire \time_counter_reg[51]_i_1_n_3 ;
  wire \time_counter_reg[51]_i_1_n_4 ;
  wire \time_counter_reg[51]_i_1_n_5 ;
  wire \time_counter_reg[51]_i_1_n_6 ;
  wire \time_counter_reg[51]_i_1_n_7 ;
  wire \time_counter_reg[55]_i_1_n_0 ;
  wire \time_counter_reg[55]_i_1_n_1 ;
  wire \time_counter_reg[55]_i_1_n_2 ;
  wire \time_counter_reg[55]_i_1_n_3 ;
  wire \time_counter_reg[55]_i_1_n_4 ;
  wire \time_counter_reg[55]_i_1_n_5 ;
  wire \time_counter_reg[55]_i_1_n_6 ;
  wire \time_counter_reg[55]_i_1_n_7 ;
  wire \time_counter_reg[59]_i_1_n_0 ;
  wire \time_counter_reg[59]_i_1_n_1 ;
  wire \time_counter_reg[59]_i_1_n_2 ;
  wire \time_counter_reg[59]_i_1_n_3 ;
  wire \time_counter_reg[59]_i_1_n_4 ;
  wire \time_counter_reg[59]_i_1_n_5 ;
  wire \time_counter_reg[59]_i_1_n_6 ;
  wire \time_counter_reg[59]_i_1_n_7 ;
  wire \time_counter_reg[63]_i_1_n_1 ;
  wire \time_counter_reg[63]_i_1_n_2 ;
  wire \time_counter_reg[63]_i_1_n_3 ;
  wire \time_counter_reg[63]_i_1_n_4 ;
  wire \time_counter_reg[63]_i_1_n_5 ;
  wire \time_counter_reg[63]_i_1_n_6 ;
  wire \time_counter_reg[63]_i_1_n_7 ;
  wire \time_counter_reg[7]_i_1_n_0 ;
  wire \time_counter_reg[7]_i_1_n_1 ;
  wire \time_counter_reg[7]_i_1_n_2 ;
  wire \time_counter_reg[7]_i_1_n_3 ;
  wire \time_counter_reg[7]_i_1_n_4 ;
  wire \time_counter_reg[7]_i_1_n_5 ;
  wire \time_counter_reg[7]_i_1_n_6 ;
  wire \time_counter_reg[7]_i_1_n_7 ;
  wire \time_in_xfifo/fifo/dest_ack ;
  wire \time_in_xfifo/fifo/dest_req ;
  wire [64:0]\time_in_xfifo/fifo/src_in ;
  wire \time_in_xfifo/fifo/src_rcv ;
  wire \time_in_xfifo/fifo/src_send ;
  wire \time_in_xfifo/fifo/src_send03_out ;
  wire \time_out_last/handshake/fifo/dest_ack ;
  wire \time_out_last/handshake/fifo/dest_req ;
  wire \time_out_last/handshake/fifo/handshake_n_1 ;
  wire [63:0]\time_out_last/handshake/fifo/src_in ;
  wire \time_out_last/handshake/fifo/src_rcv ;
  wire \time_out_last/handshake/fifo/src_send ;
  wire [63:0]\time_out_last/out_tdata ;
  wire \time_out_last/out_time_reg_n_0_[0] ;
  wire \time_out_last/out_time_reg_n_0_[10] ;
  wire \time_out_last/out_time_reg_n_0_[11] ;
  wire \time_out_last/out_time_reg_n_0_[12] ;
  wire \time_out_last/out_time_reg_n_0_[13] ;
  wire \time_out_last/out_time_reg_n_0_[14] ;
  wire \time_out_last/out_time_reg_n_0_[15] ;
  wire \time_out_last/out_time_reg_n_0_[16] ;
  wire \time_out_last/out_time_reg_n_0_[17] ;
  wire \time_out_last/out_time_reg_n_0_[18] ;
  wire \time_out_last/out_time_reg_n_0_[19] ;
  wire \time_out_last/out_time_reg_n_0_[1] ;
  wire \time_out_last/out_time_reg_n_0_[20] ;
  wire \time_out_last/out_time_reg_n_0_[21] ;
  wire \time_out_last/out_time_reg_n_0_[22] ;
  wire \time_out_last/out_time_reg_n_0_[23] ;
  wire \time_out_last/out_time_reg_n_0_[24] ;
  wire \time_out_last/out_time_reg_n_0_[25] ;
  wire \time_out_last/out_time_reg_n_0_[26] ;
  wire \time_out_last/out_time_reg_n_0_[27] ;
  wire \time_out_last/out_time_reg_n_0_[28] ;
  wire \time_out_last/out_time_reg_n_0_[29] ;
  wire \time_out_last/out_time_reg_n_0_[2] ;
  wire \time_out_last/out_time_reg_n_0_[30] ;
  wire \time_out_last/out_time_reg_n_0_[31] ;
  wire \time_out_last/out_time_reg_n_0_[32] ;
  wire \time_out_last/out_time_reg_n_0_[33] ;
  wire \time_out_last/out_time_reg_n_0_[34] ;
  wire \time_out_last/out_time_reg_n_0_[35] ;
  wire \time_out_last/out_time_reg_n_0_[36] ;
  wire \time_out_last/out_time_reg_n_0_[37] ;
  wire \time_out_last/out_time_reg_n_0_[38] ;
  wire \time_out_last/out_time_reg_n_0_[39] ;
  wire \time_out_last/out_time_reg_n_0_[3] ;
  wire \time_out_last/out_time_reg_n_0_[40] ;
  wire \time_out_last/out_time_reg_n_0_[41] ;
  wire \time_out_last/out_time_reg_n_0_[42] ;
  wire \time_out_last/out_time_reg_n_0_[43] ;
  wire \time_out_last/out_time_reg_n_0_[44] ;
  wire \time_out_last/out_time_reg_n_0_[45] ;
  wire \time_out_last/out_time_reg_n_0_[46] ;
  wire \time_out_last/out_time_reg_n_0_[47] ;
  wire \time_out_last/out_time_reg_n_0_[48] ;
  wire \time_out_last/out_time_reg_n_0_[49] ;
  wire \time_out_last/out_time_reg_n_0_[4] ;
  wire \time_out_last/out_time_reg_n_0_[50] ;
  wire \time_out_last/out_time_reg_n_0_[51] ;
  wire \time_out_last/out_time_reg_n_0_[52] ;
  wire \time_out_last/out_time_reg_n_0_[53] ;
  wire \time_out_last/out_time_reg_n_0_[54] ;
  wire \time_out_last/out_time_reg_n_0_[55] ;
  wire \time_out_last/out_time_reg_n_0_[56] ;
  wire \time_out_last/out_time_reg_n_0_[57] ;
  wire \time_out_last/out_time_reg_n_0_[58] ;
  wire \time_out_last/out_time_reg_n_0_[59] ;
  wire \time_out_last/out_time_reg_n_0_[5] ;
  wire \time_out_last/out_time_reg_n_0_[60] ;
  wire \time_out_last/out_time_reg_n_0_[61] ;
  wire \time_out_last/out_time_reg_n_0_[62] ;
  wire \time_out_last/out_time_reg_n_0_[63] ;
  wire \time_out_last/out_time_reg_n_0_[6] ;
  wire \time_out_last/out_time_reg_n_0_[7] ;
  wire \time_out_last/out_time_reg_n_0_[8] ;
  wire \time_out_last/out_time_reg_n_0_[9] ;
  wire \time_out_last/out_tvalid ;
  wire \time_out_now/handshake/fifo/dest_ack ;
  wire \time_out_now/handshake/fifo/dest_req ;
  wire \time_out_now/handshake/fifo/handshake_n_1 ;
  wire [63:0]\time_out_now/handshake/fifo/src_in ;
  wire \time_out_now/handshake/fifo/src_rcv ;
  wire \time_out_now/handshake/fifo/src_send ;
  wire [63:0]\time_out_now/out_tdata ;
  wire \time_out_now/out_tvalid ;
  wire [3:3]\NLW_time_counter_reg[63]_i_1_CO_UNCONNECTED ;

  FDRE \DATA_time_last_reg[0] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[0]),
        .Q(DATA_time_last[0]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[10] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[10]),
        .Q(DATA_time_last[10]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[11] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[11]),
        .Q(DATA_time_last[11]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[12] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[12]),
        .Q(DATA_time_last[12]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[13] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[13]),
        .Q(DATA_time_last[13]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[14] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[14]),
        .Q(DATA_time_last[14]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[15] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[15]),
        .Q(DATA_time_last[15]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[16] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[16]),
        .Q(DATA_time_last[16]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[17] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[17]),
        .Q(DATA_time_last[17]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[18] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[18]),
        .Q(DATA_time_last[18]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[19] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[19]),
        .Q(DATA_time_last[19]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[1] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[1]),
        .Q(DATA_time_last[1]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[20] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[20]),
        .Q(DATA_time_last[20]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[21] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[21]),
        .Q(DATA_time_last[21]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[22] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[22]),
        .Q(DATA_time_last[22]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[23] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[23]),
        .Q(DATA_time_last[23]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[24] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[24]),
        .Q(DATA_time_last[24]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[25] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[25]),
        .Q(DATA_time_last[25]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[26] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[26]),
        .Q(DATA_time_last[26]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[27] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[27]),
        .Q(DATA_time_last[27]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[28] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[28]),
        .Q(DATA_time_last[28]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[29] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[29]),
        .Q(DATA_time_last[29]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[2] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[2]),
        .Q(DATA_time_last[2]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[30] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[30]),
        .Q(DATA_time_last[30]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[31] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[31]),
        .Q(DATA_time_last[31]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[32] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[32]),
        .Q(DATA_time_last[32]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[33] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[33]),
        .Q(DATA_time_last[33]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[34] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[34]),
        .Q(DATA_time_last[34]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[35] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[35]),
        .Q(DATA_time_last[35]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[36] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[36]),
        .Q(DATA_time_last[36]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[37] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[37]),
        .Q(DATA_time_last[37]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[38] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[38]),
        .Q(DATA_time_last[38]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[39] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[39]),
        .Q(DATA_time_last[39]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[3] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[3]),
        .Q(DATA_time_last[3]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[40] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[40]),
        .Q(DATA_time_last[40]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[41] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[41]),
        .Q(DATA_time_last[41]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[42] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[42]),
        .Q(DATA_time_last[42]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[43] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[43]),
        .Q(DATA_time_last[43]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[44] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[44]),
        .Q(DATA_time_last[44]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[45] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[45]),
        .Q(DATA_time_last[45]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[46] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[46]),
        .Q(DATA_time_last[46]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[47] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[47]),
        .Q(DATA_time_last[47]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[48] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[48]),
        .Q(DATA_time_last[48]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[49] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[49]),
        .Q(DATA_time_last[49]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[4] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[4]),
        .Q(DATA_time_last[4]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[50] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[50]),
        .Q(DATA_time_last[50]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[51] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[51]),
        .Q(DATA_time_last[51]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[52] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[52]),
        .Q(DATA_time_last[52]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[53] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[53]),
        .Q(DATA_time_last[53]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[54] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[54]),
        .Q(DATA_time_last[54]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[55] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[55]),
        .Q(DATA_time_last[55]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[56] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[56]),
        .Q(DATA_time_last[56]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[57] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[57]),
        .Q(DATA_time_last[57]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[58] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[58]),
        .Q(DATA_time_last[58]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[59] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[59]),
        .Q(DATA_time_last[59]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[5] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[5]),
        .Q(DATA_time_last[5]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[60] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[60]),
        .Q(DATA_time_last[60]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[61] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[61]),
        .Q(DATA_time_last[61]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[62] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[62]),
        .Q(DATA_time_last[62]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[63] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[63]),
        .Q(DATA_time_last[63]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[6] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[6]),
        .Q(DATA_time_last[6]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[7] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[7]),
        .Q(DATA_time_last[7]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[8] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[8]),
        .Q(DATA_time_last[8]),
        .R(DATA_rst));
  FDRE \DATA_time_last_reg[9] 
       (.C(DATA_clk),
        .CE(DATA_trigger_in),
        .D(DATA_time_now[9]),
        .Q(DATA_time_last[9]),
        .R(DATA_rst));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[0]_i_1 
       (.I0(out_time[0]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[0] ),
        .O(\SYS_time_out[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[10]_i_1 
       (.I0(out_time[10]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[10] ),
        .O(\SYS_time_out[10]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[11]_i_1 
       (.I0(out_time[11]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[11] ),
        .O(\SYS_time_out[11]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[12]_i_1 
       (.I0(out_time[12]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[12] ),
        .O(\SYS_time_out[12]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[13]_i_1 
       (.I0(out_time[13]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[13] ),
        .O(\SYS_time_out[13]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[14]_i_1 
       (.I0(out_time[14]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[14] ),
        .O(\SYS_time_out[14]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[15]_i_1 
       (.I0(out_time[15]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[15] ),
        .O(\SYS_time_out[15]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[16]_i_1 
       (.I0(out_time[16]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[16] ),
        .O(\SYS_time_out[16]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[17]_i_1 
       (.I0(out_time[17]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[17] ),
        .O(\SYS_time_out[17]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[18]_i_1 
       (.I0(out_time[18]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[18] ),
        .O(\SYS_time_out[18]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[19]_i_1 
       (.I0(out_time[19]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[19] ),
        .O(\SYS_time_out[19]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[1]_i_1 
       (.I0(out_time[1]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[1] ),
        .O(\SYS_time_out[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[20]_i_1 
       (.I0(out_time[20]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[20] ),
        .O(\SYS_time_out[20]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[21]_i_1 
       (.I0(out_time[21]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[21] ),
        .O(\SYS_time_out[21]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[22]_i_1 
       (.I0(out_time[22]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[22] ),
        .O(\SYS_time_out[22]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[23]_i_1 
       (.I0(out_time[23]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[23] ),
        .O(\SYS_time_out[23]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[24]_i_1 
       (.I0(out_time[24]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[24] ),
        .O(\SYS_time_out[24]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[25]_i_1 
       (.I0(out_time[25]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[25] ),
        .O(\SYS_time_out[25]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[26]_i_1 
       (.I0(out_time[26]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[26] ),
        .O(\SYS_time_out[26]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[27]_i_1 
       (.I0(out_time[27]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[27] ),
        .O(\SYS_time_out[27]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[28]_i_1 
       (.I0(out_time[28]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[28] ),
        .O(\SYS_time_out[28]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[29]_i_1 
       (.I0(out_time[29]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[29] ),
        .O(\SYS_time_out[29]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[2]_i_1 
       (.I0(out_time[2]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[2] ),
        .O(\SYS_time_out[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[30]_i_1 
       (.I0(out_time[30]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[30] ),
        .O(\SYS_time_out[30]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[31]_i_1 
       (.I0(out_time[31]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[31] ),
        .O(\SYS_time_out[31]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[32]_i_1 
       (.I0(out_time[32]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[32] ),
        .O(\SYS_time_out[32]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[33]_i_1 
       (.I0(out_time[33]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[33] ),
        .O(\SYS_time_out[33]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[34]_i_1 
       (.I0(out_time[34]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[34] ),
        .O(\SYS_time_out[34]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[35]_i_1 
       (.I0(out_time[35]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[35] ),
        .O(\SYS_time_out[35]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[36]_i_1 
       (.I0(out_time[36]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[36] ),
        .O(\SYS_time_out[36]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[37]_i_1 
       (.I0(out_time[37]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[37] ),
        .O(\SYS_time_out[37]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[38]_i_1 
       (.I0(out_time[38]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[38] ),
        .O(\SYS_time_out[38]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[39]_i_1 
       (.I0(out_time[39]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[39] ),
        .O(\SYS_time_out[39]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[3]_i_1 
       (.I0(out_time[3]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[3] ),
        .O(\SYS_time_out[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[40]_i_1 
       (.I0(out_time[40]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[40] ),
        .O(\SYS_time_out[40]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[41]_i_1 
       (.I0(out_time[41]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[41] ),
        .O(\SYS_time_out[41]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[42]_i_1 
       (.I0(out_time[42]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[42] ),
        .O(\SYS_time_out[42]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[43]_i_1 
       (.I0(out_time[43]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[43] ),
        .O(\SYS_time_out[43]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[44]_i_1 
       (.I0(out_time[44]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[44] ),
        .O(\SYS_time_out[44]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[45]_i_1 
       (.I0(out_time[45]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[45] ),
        .O(\SYS_time_out[45]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[46]_i_1 
       (.I0(out_time[46]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[46] ),
        .O(\SYS_time_out[46]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[47]_i_1 
       (.I0(out_time[47]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[47] ),
        .O(\SYS_time_out[47]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[48]_i_1 
       (.I0(out_time[48]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[48] ),
        .O(\SYS_time_out[48]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[49]_i_1 
       (.I0(out_time[49]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[49] ),
        .O(\SYS_time_out[49]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[4]_i_1 
       (.I0(out_time[4]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[4] ),
        .O(\SYS_time_out[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[50]_i_1 
       (.I0(out_time[50]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[50] ),
        .O(\SYS_time_out[50]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[51]_i_1 
       (.I0(out_time[51]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[51] ),
        .O(\SYS_time_out[51]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[52]_i_1 
       (.I0(out_time[52]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[52] ),
        .O(\SYS_time_out[52]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[53]_i_1 
       (.I0(out_time[53]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[53] ),
        .O(\SYS_time_out[53]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[54]_i_1 
       (.I0(out_time[54]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[54] ),
        .O(\SYS_time_out[54]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[55]_i_1 
       (.I0(out_time[55]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[55] ),
        .O(\SYS_time_out[55]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[56]_i_1 
       (.I0(out_time[56]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[56] ),
        .O(\SYS_time_out[56]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[57]_i_1 
       (.I0(out_time[57]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[57] ),
        .O(\SYS_time_out[57]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[58]_i_1 
       (.I0(out_time[58]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[58] ),
        .O(\SYS_time_out[58]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[59]_i_1 
       (.I0(out_time[59]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[59] ),
        .O(\SYS_time_out[59]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[5]_i_1 
       (.I0(out_time[5]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[5] ),
        .O(\SYS_time_out[5]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[60]_i_1 
       (.I0(out_time[60]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[60] ),
        .O(\SYS_time_out[60]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[61]_i_1 
       (.I0(out_time[61]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[61] ),
        .O(\SYS_time_out[61]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[62]_i_1 
       (.I0(out_time[62]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[62] ),
        .O(\SYS_time_out[62]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[63]_i_1 
       (.I0(out_time[63]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[63] ),
        .O(\SYS_time_out[63]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[6]_i_1 
       (.I0(out_time[6]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[6] ),
        .O(\SYS_time_out[6]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[7]_i_1 
       (.I0(out_time[7]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[7] ),
        .O(\SYS_time_out[7]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[8]_i_1 
       (.I0(out_time[8]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[8] ),
        .O(\SYS_time_out[8]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \SYS_time_out[9]_i_1 
       (.I0(out_time[9]),
        .I1(SYS_time_asap),
        .I2(\time_out_last/out_time_reg_n_0_[9] ),
        .O(\SYS_time_out[9]_i_1_n_0 ));
  FDRE \SYS_time_out_reg[0] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[0]_i_1_n_0 ),
        .Q(SYS_time_out[0]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[10] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[10]_i_1_n_0 ),
        .Q(SYS_time_out[10]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[11] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[11]_i_1_n_0 ),
        .Q(SYS_time_out[11]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[12] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[12]_i_1_n_0 ),
        .Q(SYS_time_out[12]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[13] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[13]_i_1_n_0 ),
        .Q(SYS_time_out[13]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[14] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[14]_i_1_n_0 ),
        .Q(SYS_time_out[14]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[15] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[15]_i_1_n_0 ),
        .Q(SYS_time_out[15]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[16] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[16]_i_1_n_0 ),
        .Q(SYS_time_out[16]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[17] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[17]_i_1_n_0 ),
        .Q(SYS_time_out[17]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[18] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[18]_i_1_n_0 ),
        .Q(SYS_time_out[18]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[19] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[19]_i_1_n_0 ),
        .Q(SYS_time_out[19]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[1] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[1]_i_1_n_0 ),
        .Q(SYS_time_out[1]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[20] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[20]_i_1_n_0 ),
        .Q(SYS_time_out[20]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[21] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[21]_i_1_n_0 ),
        .Q(SYS_time_out[21]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[22] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[22]_i_1_n_0 ),
        .Q(SYS_time_out[22]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[23] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[23]_i_1_n_0 ),
        .Q(SYS_time_out[23]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[24] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[24]_i_1_n_0 ),
        .Q(SYS_time_out[24]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[25] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[25]_i_1_n_0 ),
        .Q(SYS_time_out[25]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[26] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[26]_i_1_n_0 ),
        .Q(SYS_time_out[26]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[27] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[27]_i_1_n_0 ),
        .Q(SYS_time_out[27]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[28] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[28]_i_1_n_0 ),
        .Q(SYS_time_out[28]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[29] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[29]_i_1_n_0 ),
        .Q(SYS_time_out[29]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[2] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[2]_i_1_n_0 ),
        .Q(SYS_time_out[2]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[30] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[30]_i_1_n_0 ),
        .Q(SYS_time_out[30]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[31] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[31]_i_1_n_0 ),
        .Q(SYS_time_out[31]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[32] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[32]_i_1_n_0 ),
        .Q(SYS_time_out[32]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[33] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[33]_i_1_n_0 ),
        .Q(SYS_time_out[33]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[34] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[34]_i_1_n_0 ),
        .Q(SYS_time_out[34]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[35] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[35]_i_1_n_0 ),
        .Q(SYS_time_out[35]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[36] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[36]_i_1_n_0 ),
        .Q(SYS_time_out[36]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[37] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[37]_i_1_n_0 ),
        .Q(SYS_time_out[37]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[38] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[38]_i_1_n_0 ),
        .Q(SYS_time_out[38]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[39] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[39]_i_1_n_0 ),
        .Q(SYS_time_out[39]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[3] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[3]_i_1_n_0 ),
        .Q(SYS_time_out[3]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[40] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[40]_i_1_n_0 ),
        .Q(SYS_time_out[40]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[41] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[41]_i_1_n_0 ),
        .Q(SYS_time_out[41]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[42] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[42]_i_1_n_0 ),
        .Q(SYS_time_out[42]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[43] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[43]_i_1_n_0 ),
        .Q(SYS_time_out[43]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[44] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[44]_i_1_n_0 ),
        .Q(SYS_time_out[44]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[45] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[45]_i_1_n_0 ),
        .Q(SYS_time_out[45]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[46] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[46]_i_1_n_0 ),
        .Q(SYS_time_out[46]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[47] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[47]_i_1_n_0 ),
        .Q(SYS_time_out[47]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[48] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[48]_i_1_n_0 ),
        .Q(SYS_time_out[48]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[49] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[49]_i_1_n_0 ),
        .Q(SYS_time_out[49]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[4] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[4]_i_1_n_0 ),
        .Q(SYS_time_out[4]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[50] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[50]_i_1_n_0 ),
        .Q(SYS_time_out[50]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[51] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[51]_i_1_n_0 ),
        .Q(SYS_time_out[51]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[52] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[52]_i_1_n_0 ),
        .Q(SYS_time_out[52]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[53] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[53]_i_1_n_0 ),
        .Q(SYS_time_out[53]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[54] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[54]_i_1_n_0 ),
        .Q(SYS_time_out[54]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[55] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[55]_i_1_n_0 ),
        .Q(SYS_time_out[55]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[56] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[56]_i_1_n_0 ),
        .Q(SYS_time_out[56]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[57] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[57]_i_1_n_0 ),
        .Q(SYS_time_out[57]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[58] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[58]_i_1_n_0 ),
        .Q(SYS_time_out[58]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[59] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[59]_i_1_n_0 ),
        .Q(SYS_time_out[59]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[5] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[5]_i_1_n_0 ),
        .Q(SYS_time_out[5]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[60] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[60]_i_1_n_0 ),
        .Q(SYS_time_out[60]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[61] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[61]_i_1_n_0 ),
        .Q(SYS_time_out[61]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[62] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[62]_i_1_n_0 ),
        .Q(SYS_time_out[62]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[63] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[63]_i_1_n_0 ),
        .Q(SYS_time_out[63]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[6] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[6]_i_1_n_0 ),
        .Q(SYS_time_out[6]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[7] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[7]_i_1_n_0 ),
        .Q(SYS_time_out[7]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[8] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[8]_i_1_n_0 ),
        .Q(SYS_time_out[8]),
        .R(SYS_rst));
  FDRE \SYS_time_out_reg[9] 
       (.C(SYS_clk),
        .CE(SYS_time_read),
        .D(\SYS_time_out[9]_i_1_n_0 ),
        .Q(SYS_time_out[9]),
        .R(SYS_rst));
  LUT3 #(
    .INIT(8'hC8)) 
    dest_ack_i_1
       (.I0(DATA_stb),
        .I1(\time_in_xfifo/fifo/dest_req ),
        .I2(\time_in_xfifo/fifo/dest_ack ),
        .O(dest_ack_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT2 #(
    .INIT(4'h2)) 
    dest_ack_i_1__0
       (.I0(\time_out_now/handshake/fifo/dest_req ),
        .I1(SYS_rst),
        .O(dest_ack_i_1__0_n_0));
  LUT2 #(
    .INIT(4'h2)) 
    dest_ack_i_1__1
       (.I0(\time_out_last/handshake/fifo/dest_req ),
        .I1(SYS_rst),
        .O(dest_ack_i_1__1_n_0));
  LUT6 #(
    .INIT(64'h10101010FF101010)) 
    has_next_time_i_1
       (.I0(next_time_asap_reg_n_0),
        .I1(DATA_trigger_in),
        .I2(has_next_time),
        .I3(DATA_stb),
        .I4(\time_in_xfifo/fifo/dest_req ),
        .I5(\time_in_xfifo/fifo/dest_ack ),
        .O(has_next_time_i_1_n_0));
  FDRE has_next_time_reg
       (.C(DATA_clk),
        .CE(1'b1),
        .D(has_next_time_i_1_n_0),
        .Q(has_next_time),
        .R(DATA_rst));
  LUT3 #(
    .INIT(8'h40)) 
    \next_time[63]_i_1 
       (.I0(\time_in_xfifo/fifo/dest_ack ),
        .I1(\time_in_xfifo/fifo/dest_req ),
        .I2(DATA_stb),
        .O(next_time0));
  LUT6 #(
    .INIT(64'h40004000FFBF4000)) 
    next_time_asap_i_1
       (.I0(\time_in_xfifo/fifo/dest_ack ),
        .I1(\time_in_xfifo/fifo/dest_req ),
        .I2(DATA_stb),
        .I3(DATA_time_asap),
        .I4(next_time_asap_reg_n_0),
        .I5(has_next_time),
        .O(next_time_asap_i_1_n_0));
  FDRE next_time_asap_reg
       (.C(DATA_clk),
        .CE(1'b1),
        .D(next_time_asap_i_1_n_0),
        .Q(next_time_asap_reg_n_0),
        .R(DATA_rst));
  FDRE \next_time_reg[0] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[0]),
        .Q(next_time[0]),
        .R(DATA_rst));
  FDRE \next_time_reg[10] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[10]),
        .Q(next_time[10]),
        .R(DATA_rst));
  FDRE \next_time_reg[11] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[11]),
        .Q(next_time[11]),
        .R(DATA_rst));
  FDRE \next_time_reg[12] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[12]),
        .Q(next_time[12]),
        .R(DATA_rst));
  FDRE \next_time_reg[13] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[13]),
        .Q(next_time[13]),
        .R(DATA_rst));
  FDRE \next_time_reg[14] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[14]),
        .Q(next_time[14]),
        .R(DATA_rst));
  FDRE \next_time_reg[15] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[15]),
        .Q(next_time[15]),
        .R(DATA_rst));
  FDRE \next_time_reg[16] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[16]),
        .Q(next_time[16]),
        .R(DATA_rst));
  FDRE \next_time_reg[17] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[17]),
        .Q(next_time[17]),
        .R(DATA_rst));
  FDRE \next_time_reg[18] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[18]),
        .Q(next_time[18]),
        .R(DATA_rst));
  FDRE \next_time_reg[19] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[19]),
        .Q(next_time[19]),
        .R(DATA_rst));
  FDRE \next_time_reg[1] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[1]),
        .Q(next_time[1]),
        .R(DATA_rst));
  FDRE \next_time_reg[20] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[20]),
        .Q(next_time[20]),
        .R(DATA_rst));
  FDRE \next_time_reg[21] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[21]),
        .Q(next_time[21]),
        .R(DATA_rst));
  FDRE \next_time_reg[22] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[22]),
        .Q(next_time[22]),
        .R(DATA_rst));
  FDRE \next_time_reg[23] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[23]),
        .Q(next_time[23]),
        .R(DATA_rst));
  FDRE \next_time_reg[24] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[24]),
        .Q(next_time[24]),
        .R(DATA_rst));
  FDRE \next_time_reg[25] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[25]),
        .Q(next_time[25]),
        .R(DATA_rst));
  FDRE \next_time_reg[26] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[26]),
        .Q(next_time[26]),
        .R(DATA_rst));
  FDRE \next_time_reg[27] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[27]),
        .Q(next_time[27]),
        .R(DATA_rst));
  FDRE \next_time_reg[28] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[28]),
        .Q(next_time[28]),
        .R(DATA_rst));
  FDRE \next_time_reg[29] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[29]),
        .Q(next_time[29]),
        .R(DATA_rst));
  FDRE \next_time_reg[2] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[2]),
        .Q(next_time[2]),
        .R(DATA_rst));
  FDRE \next_time_reg[30] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[30]),
        .Q(next_time[30]),
        .R(DATA_rst));
  FDRE \next_time_reg[31] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[31]),
        .Q(next_time[31]),
        .R(DATA_rst));
  FDRE \next_time_reg[32] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[32]),
        .Q(next_time[32]),
        .R(DATA_rst));
  FDRE \next_time_reg[33] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[33]),
        .Q(next_time[33]),
        .R(DATA_rst));
  FDRE \next_time_reg[34] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[34]),
        .Q(next_time[34]),
        .R(DATA_rst));
  FDRE \next_time_reg[35] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[35]),
        .Q(next_time[35]),
        .R(DATA_rst));
  FDRE \next_time_reg[36] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[36]),
        .Q(next_time[36]),
        .R(DATA_rst));
  FDRE \next_time_reg[37] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[37]),
        .Q(next_time[37]),
        .R(DATA_rst));
  FDRE \next_time_reg[38] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[38]),
        .Q(next_time[38]),
        .R(DATA_rst));
  FDRE \next_time_reg[39] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[39]),
        .Q(next_time[39]),
        .R(DATA_rst));
  FDRE \next_time_reg[3] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[3]),
        .Q(next_time[3]),
        .R(DATA_rst));
  FDRE \next_time_reg[40] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[40]),
        .Q(next_time[40]),
        .R(DATA_rst));
  FDRE \next_time_reg[41] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[41]),
        .Q(next_time[41]),
        .R(DATA_rst));
  FDRE \next_time_reg[42] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[42]),
        .Q(next_time[42]),
        .R(DATA_rst));
  FDRE \next_time_reg[43] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[43]),
        .Q(next_time[43]),
        .R(DATA_rst));
  FDRE \next_time_reg[44] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[44]),
        .Q(next_time[44]),
        .R(DATA_rst));
  FDRE \next_time_reg[45] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[45]),
        .Q(next_time[45]),
        .R(DATA_rst));
  FDRE \next_time_reg[46] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[46]),
        .Q(next_time[46]),
        .R(DATA_rst));
  FDRE \next_time_reg[47] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[47]),
        .Q(next_time[47]),
        .R(DATA_rst));
  FDRE \next_time_reg[48] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[48]),
        .Q(next_time[48]),
        .R(DATA_rst));
  FDRE \next_time_reg[49] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[49]),
        .Q(next_time[49]),
        .R(DATA_rst));
  FDRE \next_time_reg[4] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[4]),
        .Q(next_time[4]),
        .R(DATA_rst));
  FDRE \next_time_reg[50] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[50]),
        .Q(next_time[50]),
        .R(DATA_rst));
  FDRE \next_time_reg[51] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[51]),
        .Q(next_time[51]),
        .R(DATA_rst));
  FDRE \next_time_reg[52] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[52]),
        .Q(next_time[52]),
        .R(DATA_rst));
  FDRE \next_time_reg[53] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[53]),
        .Q(next_time[53]),
        .R(DATA_rst));
  FDRE \next_time_reg[54] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[54]),
        .Q(next_time[54]),
        .R(DATA_rst));
  FDRE \next_time_reg[55] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[55]),
        .Q(next_time[55]),
        .R(DATA_rst));
  FDRE \next_time_reg[56] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[56]),
        .Q(next_time[56]),
        .R(DATA_rst));
  FDRE \next_time_reg[57] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[57]),
        .Q(next_time[57]),
        .R(DATA_rst));
  FDRE \next_time_reg[58] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[58]),
        .Q(next_time[58]),
        .R(DATA_rst));
  FDRE \next_time_reg[59] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[59]),
        .Q(next_time[59]),
        .R(DATA_rst));
  FDRE \next_time_reg[5] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[5]),
        .Q(next_time[5]),
        .R(DATA_rst));
  FDRE \next_time_reg[60] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[60]),
        .Q(next_time[60]),
        .R(DATA_rst));
  FDRE \next_time_reg[61] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[61]),
        .Q(next_time[61]),
        .R(DATA_rst));
  FDRE \next_time_reg[62] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[62]),
        .Q(next_time[62]),
        .R(DATA_rst));
  FDRE \next_time_reg[63] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[63]),
        .Q(next_time[63]),
        .R(DATA_rst));
  FDRE \next_time_reg[6] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[6]),
        .Q(next_time[6]),
        .R(DATA_rst));
  FDRE \next_time_reg[7] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[7]),
        .Q(next_time[7]),
        .R(DATA_rst));
  FDRE \next_time_reg[8] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[8]),
        .Q(next_time[8]),
        .R(DATA_rst));
  FDRE \next_time_reg[9] 
       (.C(DATA_clk),
        .CE(next_time0),
        .D(DATA_time_next[9]),
        .Q(next_time[9]),
        .R(DATA_rst));
  LUT2 #(
    .INIT(4'h2)) 
    \out_time[63]_i_1 
       (.I0(\time_out_now/handshake/fifo/dest_req ),
        .I1(\time_out_now/handshake/fifo/dest_ack ),
        .O(\time_out_now/out_tvalid ));
  LUT2 #(
    .INIT(4'h2)) 
    \out_time[63]_i_1__0 
       (.I0(\time_out_last/handshake/fifo/dest_req ),
        .I1(\time_out_last/handshake/fifo/dest_ack ),
        .O(\time_out_last/out_tvalid ));
  LUT2 #(
    .INIT(4'h1)) 
    \src_in[63]_i_1 
       (.I0(\time_out_now/handshake/fifo/src_send ),
        .I1(\time_out_now/handshake/fifo/src_rcv ),
        .O(\src_in[63]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h1)) 
    \src_in[63]_i_1__0 
       (.I0(\time_out_last/handshake/fifo/src_send ),
        .I1(\time_out_last/handshake/fifo/src_rcv ),
        .O(\src_in[63]_i_1__0_n_0 ));
  LUT3 #(
    .INIT(8'h02)) 
    \src_in[64]_i_1 
       (.I0(SYS_time_write),
        .I1(\time_in_xfifo/fifo/src_rcv ),
        .I2(\time_in_xfifo/fifo/src_send ),
        .O(\time_in_xfifo/fifo/src_send03_out ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT4 #(
    .INIT(16'h0032)) 
    src_send_i_1
       (.I0(\time_in_xfifo/fifo/src_send ),
        .I1(\time_in_xfifo/fifo/src_rcv ),
        .I2(SYS_time_write),
        .I3(SYS_rst),
        .O(src_send_i_1_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    src_send_i_1__0
       (.I0(\time_out_now/handshake/fifo/src_rcv ),
        .O(src_send_i_1__0_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    src_send_i_1__1
       (.I0(\time_out_last/handshake/fifo/src_rcv ),
        .O(src_send_i_1__1_n_0));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[11]_i_2 
       (.I0(next_time[11]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[11]),
        .O(\time_counter[11]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[11]_i_3 
       (.I0(next_time[10]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[10]),
        .O(\time_counter[11]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[11]_i_4 
       (.I0(next_time[9]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[9]),
        .O(\time_counter[11]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[11]_i_5 
       (.I0(next_time[8]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[8]),
        .O(\time_counter[11]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[15]_i_2 
       (.I0(next_time[15]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[15]),
        .O(\time_counter[15]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[15]_i_3 
       (.I0(next_time[14]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[14]),
        .O(\time_counter[15]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[15]_i_4 
       (.I0(next_time[13]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[13]),
        .O(\time_counter[15]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[15]_i_5 
       (.I0(next_time[12]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[12]),
        .O(\time_counter[15]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[19]_i_2 
       (.I0(next_time[19]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[19]),
        .O(\time_counter[19]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[19]_i_3 
       (.I0(next_time[18]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[18]),
        .O(\time_counter[19]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[19]_i_4 
       (.I0(next_time[17]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[17]),
        .O(\time_counter[19]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[19]_i_5 
       (.I0(next_time[16]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[16]),
        .O(\time_counter[19]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[23]_i_2 
       (.I0(next_time[23]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[23]),
        .O(\time_counter[23]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[23]_i_3 
       (.I0(next_time[22]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[22]),
        .O(\time_counter[23]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[23]_i_4 
       (.I0(next_time[21]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[21]),
        .O(\time_counter[23]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[23]_i_5 
       (.I0(next_time[20]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[20]),
        .O(\time_counter[23]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[27]_i_2 
       (.I0(next_time[27]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[27]),
        .O(\time_counter[27]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[27]_i_3 
       (.I0(next_time[26]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[26]),
        .O(\time_counter[27]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[27]_i_4 
       (.I0(next_time[25]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[25]),
        .O(\time_counter[27]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[27]_i_5 
       (.I0(next_time[24]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[24]),
        .O(\time_counter[27]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[31]_i_2 
       (.I0(next_time[31]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[31]),
        .O(\time_counter[31]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[31]_i_3 
       (.I0(next_time[30]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[30]),
        .O(\time_counter[31]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[31]_i_4 
       (.I0(next_time[29]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[29]),
        .O(\time_counter[31]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[31]_i_5 
       (.I0(next_time[28]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[28]),
        .O(\time_counter[31]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[35]_i_2 
       (.I0(next_time[35]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[35]),
        .O(\time_counter[35]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[35]_i_3 
       (.I0(next_time[34]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[34]),
        .O(\time_counter[35]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[35]_i_4 
       (.I0(next_time[33]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[33]),
        .O(\time_counter[35]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[35]_i_5 
       (.I0(next_time[32]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[32]),
        .O(\time_counter[35]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[39]_i_2 
       (.I0(next_time[39]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[39]),
        .O(\time_counter[39]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[39]_i_3 
       (.I0(next_time[38]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[38]),
        .O(\time_counter[39]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[39]_i_4 
       (.I0(next_time[37]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[37]),
        .O(\time_counter[39]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[39]_i_5 
       (.I0(next_time[36]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[36]),
        .O(\time_counter[39]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[3]_i_2 
       (.I0(next_time[0]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[0]),
        .O(\time_counter[3]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[3]_i_3 
       (.I0(next_time[3]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[3]),
        .O(\time_counter[3]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[3]_i_4 
       (.I0(next_time[2]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[2]),
        .O(\time_counter[3]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[3]_i_5 
       (.I0(next_time[1]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[1]),
        .O(\time_counter[3]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hCCC55555)) 
    \time_counter[3]_i_6 
       (.I0(DATA_time_now[0]),
        .I1(next_time[0]),
        .I2(next_time_asap_reg_n_0),
        .I3(DATA_trigger_in),
        .I4(has_next_time),
        .O(\time_counter[3]_i_6_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[43]_i_2 
       (.I0(next_time[43]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[43]),
        .O(\time_counter[43]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[43]_i_3 
       (.I0(next_time[42]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[42]),
        .O(\time_counter[43]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[43]_i_4 
       (.I0(next_time[41]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[41]),
        .O(\time_counter[43]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[43]_i_5 
       (.I0(next_time[40]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[40]),
        .O(\time_counter[43]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[47]_i_2 
       (.I0(next_time[47]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[47]),
        .O(\time_counter[47]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[47]_i_3 
       (.I0(next_time[46]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[46]),
        .O(\time_counter[47]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[47]_i_4 
       (.I0(next_time[45]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[45]),
        .O(\time_counter[47]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[47]_i_5 
       (.I0(next_time[44]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[44]),
        .O(\time_counter[47]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[51]_i_2 
       (.I0(next_time[51]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[51]),
        .O(\time_counter[51]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[51]_i_3 
       (.I0(next_time[50]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[50]),
        .O(\time_counter[51]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[51]_i_4 
       (.I0(next_time[49]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[49]),
        .O(\time_counter[51]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[51]_i_5 
       (.I0(next_time[48]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[48]),
        .O(\time_counter[51]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[55]_i_2 
       (.I0(next_time[55]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[55]),
        .O(\time_counter[55]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[55]_i_3 
       (.I0(next_time[54]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[54]),
        .O(\time_counter[55]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[55]_i_4 
       (.I0(next_time[53]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[53]),
        .O(\time_counter[55]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[55]_i_5 
       (.I0(next_time[52]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[52]),
        .O(\time_counter[55]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[59]_i_2 
       (.I0(next_time[59]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[59]),
        .O(\time_counter[59]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[59]_i_3 
       (.I0(next_time[58]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[58]),
        .O(\time_counter[59]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[59]_i_4 
       (.I0(next_time[57]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[57]),
        .O(\time_counter[59]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[59]_i_5 
       (.I0(next_time[56]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[56]),
        .O(\time_counter[59]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[63]_i_2 
       (.I0(next_time[63]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[63]),
        .O(\time_counter[63]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[63]_i_3 
       (.I0(next_time[62]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[62]),
        .O(\time_counter[63]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[63]_i_4 
       (.I0(next_time[61]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[61]),
        .O(\time_counter[63]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[63]_i_5 
       (.I0(next_time[60]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[60]),
        .O(\time_counter[63]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[7]_i_2 
       (.I0(next_time[7]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[7]),
        .O(\time_counter[7]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[7]_i_3 
       (.I0(next_time[6]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[6]),
        .O(\time_counter[7]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[7]_i_4 
       (.I0(next_time[5]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[5]),
        .O(\time_counter[7]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hBBBF8880)) 
    \time_counter[7]_i_5 
       (.I0(next_time[4]),
        .I1(has_next_time),
        .I2(DATA_trigger_in),
        .I3(next_time_asap_reg_n_0),
        .I4(DATA_time_now[4]),
        .O(\time_counter[7]_i_5_n_0 ));
  FDRE \time_counter_reg[0] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[3]_i_1_n_7 ),
        .Q(DATA_time_now[0]),
        .R(DATA_rst));
  FDRE \time_counter_reg[10] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[11]_i_1_n_5 ),
        .Q(DATA_time_now[10]),
        .R(DATA_rst));
  FDRE \time_counter_reg[11] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[11]_i_1_n_4 ),
        .Q(DATA_time_now[11]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[11]_i_1 
       (.CI(\time_counter_reg[7]_i_1_n_0 ),
        .CO({\time_counter_reg[11]_i_1_n_0 ,\time_counter_reg[11]_i_1_n_1 ,\time_counter_reg[11]_i_1_n_2 ,\time_counter_reg[11]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[11]_i_1_n_4 ,\time_counter_reg[11]_i_1_n_5 ,\time_counter_reg[11]_i_1_n_6 ,\time_counter_reg[11]_i_1_n_7 }),
        .S({\time_counter[11]_i_2_n_0 ,\time_counter[11]_i_3_n_0 ,\time_counter[11]_i_4_n_0 ,\time_counter[11]_i_5_n_0 }));
  FDRE \time_counter_reg[12] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[15]_i_1_n_7 ),
        .Q(DATA_time_now[12]),
        .R(DATA_rst));
  FDRE \time_counter_reg[13] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[15]_i_1_n_6 ),
        .Q(DATA_time_now[13]),
        .R(DATA_rst));
  FDRE \time_counter_reg[14] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[15]_i_1_n_5 ),
        .Q(DATA_time_now[14]),
        .R(DATA_rst));
  FDRE \time_counter_reg[15] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[15]_i_1_n_4 ),
        .Q(DATA_time_now[15]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[15]_i_1 
       (.CI(\time_counter_reg[11]_i_1_n_0 ),
        .CO({\time_counter_reg[15]_i_1_n_0 ,\time_counter_reg[15]_i_1_n_1 ,\time_counter_reg[15]_i_1_n_2 ,\time_counter_reg[15]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[15]_i_1_n_4 ,\time_counter_reg[15]_i_1_n_5 ,\time_counter_reg[15]_i_1_n_6 ,\time_counter_reg[15]_i_1_n_7 }),
        .S({\time_counter[15]_i_2_n_0 ,\time_counter[15]_i_3_n_0 ,\time_counter[15]_i_4_n_0 ,\time_counter[15]_i_5_n_0 }));
  FDRE \time_counter_reg[16] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[19]_i_1_n_7 ),
        .Q(DATA_time_now[16]),
        .R(DATA_rst));
  FDRE \time_counter_reg[17] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[19]_i_1_n_6 ),
        .Q(DATA_time_now[17]),
        .R(DATA_rst));
  FDRE \time_counter_reg[18] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[19]_i_1_n_5 ),
        .Q(DATA_time_now[18]),
        .R(DATA_rst));
  FDRE \time_counter_reg[19] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[19]_i_1_n_4 ),
        .Q(DATA_time_now[19]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[19]_i_1 
       (.CI(\time_counter_reg[15]_i_1_n_0 ),
        .CO({\time_counter_reg[19]_i_1_n_0 ,\time_counter_reg[19]_i_1_n_1 ,\time_counter_reg[19]_i_1_n_2 ,\time_counter_reg[19]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[19]_i_1_n_4 ,\time_counter_reg[19]_i_1_n_5 ,\time_counter_reg[19]_i_1_n_6 ,\time_counter_reg[19]_i_1_n_7 }),
        .S({\time_counter[19]_i_2_n_0 ,\time_counter[19]_i_3_n_0 ,\time_counter[19]_i_4_n_0 ,\time_counter[19]_i_5_n_0 }));
  FDRE \time_counter_reg[1] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[3]_i_1_n_6 ),
        .Q(DATA_time_now[1]),
        .R(DATA_rst));
  FDRE \time_counter_reg[20] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[23]_i_1_n_7 ),
        .Q(DATA_time_now[20]),
        .R(DATA_rst));
  FDRE \time_counter_reg[21] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[23]_i_1_n_6 ),
        .Q(DATA_time_now[21]),
        .R(DATA_rst));
  FDRE \time_counter_reg[22] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[23]_i_1_n_5 ),
        .Q(DATA_time_now[22]),
        .R(DATA_rst));
  FDRE \time_counter_reg[23] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[23]_i_1_n_4 ),
        .Q(DATA_time_now[23]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[23]_i_1 
       (.CI(\time_counter_reg[19]_i_1_n_0 ),
        .CO({\time_counter_reg[23]_i_1_n_0 ,\time_counter_reg[23]_i_1_n_1 ,\time_counter_reg[23]_i_1_n_2 ,\time_counter_reg[23]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[23]_i_1_n_4 ,\time_counter_reg[23]_i_1_n_5 ,\time_counter_reg[23]_i_1_n_6 ,\time_counter_reg[23]_i_1_n_7 }),
        .S({\time_counter[23]_i_2_n_0 ,\time_counter[23]_i_3_n_0 ,\time_counter[23]_i_4_n_0 ,\time_counter[23]_i_5_n_0 }));
  FDRE \time_counter_reg[24] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[27]_i_1_n_7 ),
        .Q(DATA_time_now[24]),
        .R(DATA_rst));
  FDRE \time_counter_reg[25] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[27]_i_1_n_6 ),
        .Q(DATA_time_now[25]),
        .R(DATA_rst));
  FDRE \time_counter_reg[26] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[27]_i_1_n_5 ),
        .Q(DATA_time_now[26]),
        .R(DATA_rst));
  FDRE \time_counter_reg[27] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[27]_i_1_n_4 ),
        .Q(DATA_time_now[27]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[27]_i_1 
       (.CI(\time_counter_reg[23]_i_1_n_0 ),
        .CO({\time_counter_reg[27]_i_1_n_0 ,\time_counter_reg[27]_i_1_n_1 ,\time_counter_reg[27]_i_1_n_2 ,\time_counter_reg[27]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[27]_i_1_n_4 ,\time_counter_reg[27]_i_1_n_5 ,\time_counter_reg[27]_i_1_n_6 ,\time_counter_reg[27]_i_1_n_7 }),
        .S({\time_counter[27]_i_2_n_0 ,\time_counter[27]_i_3_n_0 ,\time_counter[27]_i_4_n_0 ,\time_counter[27]_i_5_n_0 }));
  FDRE \time_counter_reg[28] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[31]_i_1_n_7 ),
        .Q(DATA_time_now[28]),
        .R(DATA_rst));
  FDRE \time_counter_reg[29] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[31]_i_1_n_6 ),
        .Q(DATA_time_now[29]),
        .R(DATA_rst));
  FDRE \time_counter_reg[2] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[3]_i_1_n_5 ),
        .Q(DATA_time_now[2]),
        .R(DATA_rst));
  FDRE \time_counter_reg[30] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[31]_i_1_n_5 ),
        .Q(DATA_time_now[30]),
        .R(DATA_rst));
  FDRE \time_counter_reg[31] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[31]_i_1_n_4 ),
        .Q(DATA_time_now[31]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[31]_i_1 
       (.CI(\time_counter_reg[27]_i_1_n_0 ),
        .CO({\time_counter_reg[31]_i_1_n_0 ,\time_counter_reg[31]_i_1_n_1 ,\time_counter_reg[31]_i_1_n_2 ,\time_counter_reg[31]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[31]_i_1_n_4 ,\time_counter_reg[31]_i_1_n_5 ,\time_counter_reg[31]_i_1_n_6 ,\time_counter_reg[31]_i_1_n_7 }),
        .S({\time_counter[31]_i_2_n_0 ,\time_counter[31]_i_3_n_0 ,\time_counter[31]_i_4_n_0 ,\time_counter[31]_i_5_n_0 }));
  FDRE \time_counter_reg[32] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[35]_i_1_n_7 ),
        .Q(DATA_time_now[32]),
        .R(DATA_rst));
  FDRE \time_counter_reg[33] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[35]_i_1_n_6 ),
        .Q(DATA_time_now[33]),
        .R(DATA_rst));
  FDRE \time_counter_reg[34] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[35]_i_1_n_5 ),
        .Q(DATA_time_now[34]),
        .R(DATA_rst));
  FDRE \time_counter_reg[35] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[35]_i_1_n_4 ),
        .Q(DATA_time_now[35]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[35]_i_1 
       (.CI(\time_counter_reg[31]_i_1_n_0 ),
        .CO({\time_counter_reg[35]_i_1_n_0 ,\time_counter_reg[35]_i_1_n_1 ,\time_counter_reg[35]_i_1_n_2 ,\time_counter_reg[35]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[35]_i_1_n_4 ,\time_counter_reg[35]_i_1_n_5 ,\time_counter_reg[35]_i_1_n_6 ,\time_counter_reg[35]_i_1_n_7 }),
        .S({\time_counter[35]_i_2_n_0 ,\time_counter[35]_i_3_n_0 ,\time_counter[35]_i_4_n_0 ,\time_counter[35]_i_5_n_0 }));
  FDRE \time_counter_reg[36] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[39]_i_1_n_7 ),
        .Q(DATA_time_now[36]),
        .R(DATA_rst));
  FDRE \time_counter_reg[37] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[39]_i_1_n_6 ),
        .Q(DATA_time_now[37]),
        .R(DATA_rst));
  FDRE \time_counter_reg[38] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[39]_i_1_n_5 ),
        .Q(DATA_time_now[38]),
        .R(DATA_rst));
  FDRE \time_counter_reg[39] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[39]_i_1_n_4 ),
        .Q(DATA_time_now[39]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[39]_i_1 
       (.CI(\time_counter_reg[35]_i_1_n_0 ),
        .CO({\time_counter_reg[39]_i_1_n_0 ,\time_counter_reg[39]_i_1_n_1 ,\time_counter_reg[39]_i_1_n_2 ,\time_counter_reg[39]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[39]_i_1_n_4 ,\time_counter_reg[39]_i_1_n_5 ,\time_counter_reg[39]_i_1_n_6 ,\time_counter_reg[39]_i_1_n_7 }),
        .S({\time_counter[39]_i_2_n_0 ,\time_counter[39]_i_3_n_0 ,\time_counter[39]_i_4_n_0 ,\time_counter[39]_i_5_n_0 }));
  FDRE \time_counter_reg[3] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[3]_i_1_n_4 ),
        .Q(DATA_time_now[3]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[3]_i_1 
       (.CI(1'b0),
        .CO({\time_counter_reg[3]_i_1_n_0 ,\time_counter_reg[3]_i_1_n_1 ,\time_counter_reg[3]_i_1_n_2 ,\time_counter_reg[3]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,\time_counter[3]_i_2_n_0 }),
        .O({\time_counter_reg[3]_i_1_n_4 ,\time_counter_reg[3]_i_1_n_5 ,\time_counter_reg[3]_i_1_n_6 ,\time_counter_reg[3]_i_1_n_7 }),
        .S({\time_counter[3]_i_3_n_0 ,\time_counter[3]_i_4_n_0 ,\time_counter[3]_i_5_n_0 ,\time_counter[3]_i_6_n_0 }));
  FDRE \time_counter_reg[40] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[43]_i_1_n_7 ),
        .Q(DATA_time_now[40]),
        .R(DATA_rst));
  FDRE \time_counter_reg[41] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[43]_i_1_n_6 ),
        .Q(DATA_time_now[41]),
        .R(DATA_rst));
  FDRE \time_counter_reg[42] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[43]_i_1_n_5 ),
        .Q(DATA_time_now[42]),
        .R(DATA_rst));
  FDRE \time_counter_reg[43] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[43]_i_1_n_4 ),
        .Q(DATA_time_now[43]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[43]_i_1 
       (.CI(\time_counter_reg[39]_i_1_n_0 ),
        .CO({\time_counter_reg[43]_i_1_n_0 ,\time_counter_reg[43]_i_1_n_1 ,\time_counter_reg[43]_i_1_n_2 ,\time_counter_reg[43]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[43]_i_1_n_4 ,\time_counter_reg[43]_i_1_n_5 ,\time_counter_reg[43]_i_1_n_6 ,\time_counter_reg[43]_i_1_n_7 }),
        .S({\time_counter[43]_i_2_n_0 ,\time_counter[43]_i_3_n_0 ,\time_counter[43]_i_4_n_0 ,\time_counter[43]_i_5_n_0 }));
  FDRE \time_counter_reg[44] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[47]_i_1_n_7 ),
        .Q(DATA_time_now[44]),
        .R(DATA_rst));
  FDRE \time_counter_reg[45] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[47]_i_1_n_6 ),
        .Q(DATA_time_now[45]),
        .R(DATA_rst));
  FDRE \time_counter_reg[46] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[47]_i_1_n_5 ),
        .Q(DATA_time_now[46]),
        .R(DATA_rst));
  FDRE \time_counter_reg[47] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[47]_i_1_n_4 ),
        .Q(DATA_time_now[47]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[47]_i_1 
       (.CI(\time_counter_reg[43]_i_1_n_0 ),
        .CO({\time_counter_reg[47]_i_1_n_0 ,\time_counter_reg[47]_i_1_n_1 ,\time_counter_reg[47]_i_1_n_2 ,\time_counter_reg[47]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[47]_i_1_n_4 ,\time_counter_reg[47]_i_1_n_5 ,\time_counter_reg[47]_i_1_n_6 ,\time_counter_reg[47]_i_1_n_7 }),
        .S({\time_counter[47]_i_2_n_0 ,\time_counter[47]_i_3_n_0 ,\time_counter[47]_i_4_n_0 ,\time_counter[47]_i_5_n_0 }));
  FDRE \time_counter_reg[48] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[51]_i_1_n_7 ),
        .Q(DATA_time_now[48]),
        .R(DATA_rst));
  FDRE \time_counter_reg[49] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[51]_i_1_n_6 ),
        .Q(DATA_time_now[49]),
        .R(DATA_rst));
  FDRE \time_counter_reg[4] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[7]_i_1_n_7 ),
        .Q(DATA_time_now[4]),
        .R(DATA_rst));
  FDRE \time_counter_reg[50] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[51]_i_1_n_5 ),
        .Q(DATA_time_now[50]),
        .R(DATA_rst));
  FDRE \time_counter_reg[51] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[51]_i_1_n_4 ),
        .Q(DATA_time_now[51]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[51]_i_1 
       (.CI(\time_counter_reg[47]_i_1_n_0 ),
        .CO({\time_counter_reg[51]_i_1_n_0 ,\time_counter_reg[51]_i_1_n_1 ,\time_counter_reg[51]_i_1_n_2 ,\time_counter_reg[51]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[51]_i_1_n_4 ,\time_counter_reg[51]_i_1_n_5 ,\time_counter_reg[51]_i_1_n_6 ,\time_counter_reg[51]_i_1_n_7 }),
        .S({\time_counter[51]_i_2_n_0 ,\time_counter[51]_i_3_n_0 ,\time_counter[51]_i_4_n_0 ,\time_counter[51]_i_5_n_0 }));
  FDRE \time_counter_reg[52] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[55]_i_1_n_7 ),
        .Q(DATA_time_now[52]),
        .R(DATA_rst));
  FDRE \time_counter_reg[53] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[55]_i_1_n_6 ),
        .Q(DATA_time_now[53]),
        .R(DATA_rst));
  FDRE \time_counter_reg[54] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[55]_i_1_n_5 ),
        .Q(DATA_time_now[54]),
        .R(DATA_rst));
  FDRE \time_counter_reg[55] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[55]_i_1_n_4 ),
        .Q(DATA_time_now[55]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[55]_i_1 
       (.CI(\time_counter_reg[51]_i_1_n_0 ),
        .CO({\time_counter_reg[55]_i_1_n_0 ,\time_counter_reg[55]_i_1_n_1 ,\time_counter_reg[55]_i_1_n_2 ,\time_counter_reg[55]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[55]_i_1_n_4 ,\time_counter_reg[55]_i_1_n_5 ,\time_counter_reg[55]_i_1_n_6 ,\time_counter_reg[55]_i_1_n_7 }),
        .S({\time_counter[55]_i_2_n_0 ,\time_counter[55]_i_3_n_0 ,\time_counter[55]_i_4_n_0 ,\time_counter[55]_i_5_n_0 }));
  FDRE \time_counter_reg[56] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[59]_i_1_n_7 ),
        .Q(DATA_time_now[56]),
        .R(DATA_rst));
  FDRE \time_counter_reg[57] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[59]_i_1_n_6 ),
        .Q(DATA_time_now[57]),
        .R(DATA_rst));
  FDRE \time_counter_reg[58] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[59]_i_1_n_5 ),
        .Q(DATA_time_now[58]),
        .R(DATA_rst));
  FDRE \time_counter_reg[59] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[59]_i_1_n_4 ),
        .Q(DATA_time_now[59]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[59]_i_1 
       (.CI(\time_counter_reg[55]_i_1_n_0 ),
        .CO({\time_counter_reg[59]_i_1_n_0 ,\time_counter_reg[59]_i_1_n_1 ,\time_counter_reg[59]_i_1_n_2 ,\time_counter_reg[59]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[59]_i_1_n_4 ,\time_counter_reg[59]_i_1_n_5 ,\time_counter_reg[59]_i_1_n_6 ,\time_counter_reg[59]_i_1_n_7 }),
        .S({\time_counter[59]_i_2_n_0 ,\time_counter[59]_i_3_n_0 ,\time_counter[59]_i_4_n_0 ,\time_counter[59]_i_5_n_0 }));
  FDRE \time_counter_reg[5] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[7]_i_1_n_6 ),
        .Q(DATA_time_now[5]),
        .R(DATA_rst));
  FDRE \time_counter_reg[60] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[63]_i_1_n_7 ),
        .Q(DATA_time_now[60]),
        .R(DATA_rst));
  FDRE \time_counter_reg[61] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[63]_i_1_n_6 ),
        .Q(DATA_time_now[61]),
        .R(DATA_rst));
  FDRE \time_counter_reg[62] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[63]_i_1_n_5 ),
        .Q(DATA_time_now[62]),
        .R(DATA_rst));
  FDRE \time_counter_reg[63] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[63]_i_1_n_4 ),
        .Q(DATA_time_now[63]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[63]_i_1 
       (.CI(\time_counter_reg[59]_i_1_n_0 ),
        .CO({\NLW_time_counter_reg[63]_i_1_CO_UNCONNECTED [3],\time_counter_reg[63]_i_1_n_1 ,\time_counter_reg[63]_i_1_n_2 ,\time_counter_reg[63]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[63]_i_1_n_4 ,\time_counter_reg[63]_i_1_n_5 ,\time_counter_reg[63]_i_1_n_6 ,\time_counter_reg[63]_i_1_n_7 }),
        .S({\time_counter[63]_i_2_n_0 ,\time_counter[63]_i_3_n_0 ,\time_counter[63]_i_4_n_0 ,\time_counter[63]_i_5_n_0 }));
  FDRE \time_counter_reg[6] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[7]_i_1_n_5 ),
        .Q(DATA_time_now[6]),
        .R(DATA_rst));
  FDRE \time_counter_reg[7] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[7]_i_1_n_4 ),
        .Q(DATA_time_now[7]),
        .R(DATA_rst));
  CARRY4 \time_counter_reg[7]_i_1 
       (.CI(\time_counter_reg[3]_i_1_n_0 ),
        .CO({\time_counter_reg[7]_i_1_n_0 ,\time_counter_reg[7]_i_1_n_1 ,\time_counter_reg[7]_i_1_n_2 ,\time_counter_reg[7]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\time_counter_reg[7]_i_1_n_4 ,\time_counter_reg[7]_i_1_n_5 ,\time_counter_reg[7]_i_1_n_6 ,\time_counter_reg[7]_i_1_n_7 }),
        .S({\time_counter[7]_i_2_n_0 ,\time_counter[7]_i_3_n_0 ,\time_counter[7]_i_4_n_0 ,\time_counter[7]_i_5_n_0 }));
  FDRE \time_counter_reg[8] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[11]_i_1_n_7 ),
        .Q(DATA_time_now[8]),
        .R(DATA_rst));
  FDRE \time_counter_reg[9] 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(\time_counter_reg[11]_i_1_n_6 ),
        .Q(DATA_time_now[9]),
        .R(DATA_rst));
  FDRE \time_in_xfifo/fifo/dest_ack_reg 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(dest_ack_i_1_n_0),
        .Q(\time_in_xfifo/fifo/dest_ack ),
        .R(DATA_rst));
  (* DEST_EXT_HSK = "1" *) 
  (* DEST_SYNC_FF = "2" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_SYNC_FF = "2" *) 
  (* VERSION = "0" *) 
  (* WIDTH = "65" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_handshake__xdcDup__1 \time_in_xfifo/fifo/handshake 
       (.dest_ack(\time_in_xfifo/fifo/dest_ack ),
        .dest_clk(DATA_clk),
        .dest_out({DATA_time_asap,DATA_time_next}),
        .dest_req(\time_in_xfifo/fifo/dest_req ),
        .src_clk(SYS_clk),
        .src_in(\time_in_xfifo/fifo/src_in ),
        .src_rcv(\time_in_xfifo/fifo/src_rcv ),
        .src_send(\time_in_xfifo/fifo/src_send ));
  FDRE \time_in_xfifo/fifo/src_in_reg[0] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[0]),
        .Q(\time_in_xfifo/fifo/src_in [0]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[10] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[10]),
        .Q(\time_in_xfifo/fifo/src_in [10]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[11] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[11]),
        .Q(\time_in_xfifo/fifo/src_in [11]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[12] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[12]),
        .Q(\time_in_xfifo/fifo/src_in [12]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[13] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[13]),
        .Q(\time_in_xfifo/fifo/src_in [13]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[14] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[14]),
        .Q(\time_in_xfifo/fifo/src_in [14]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[15] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[15]),
        .Q(\time_in_xfifo/fifo/src_in [15]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[16] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[16]),
        .Q(\time_in_xfifo/fifo/src_in [16]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[17] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[17]),
        .Q(\time_in_xfifo/fifo/src_in [17]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[18] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[18]),
        .Q(\time_in_xfifo/fifo/src_in [18]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[19] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[19]),
        .Q(\time_in_xfifo/fifo/src_in [19]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[1] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[1]),
        .Q(\time_in_xfifo/fifo/src_in [1]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[20] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[20]),
        .Q(\time_in_xfifo/fifo/src_in [20]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[21] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[21]),
        .Q(\time_in_xfifo/fifo/src_in [21]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[22] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[22]),
        .Q(\time_in_xfifo/fifo/src_in [22]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[23] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[23]),
        .Q(\time_in_xfifo/fifo/src_in [23]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[24] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[24]),
        .Q(\time_in_xfifo/fifo/src_in [24]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[25] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[25]),
        .Q(\time_in_xfifo/fifo/src_in [25]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[26] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[26]),
        .Q(\time_in_xfifo/fifo/src_in [26]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[27] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[27]),
        .Q(\time_in_xfifo/fifo/src_in [27]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[28] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[28]),
        .Q(\time_in_xfifo/fifo/src_in [28]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[29] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[29]),
        .Q(\time_in_xfifo/fifo/src_in [29]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[2] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[2]),
        .Q(\time_in_xfifo/fifo/src_in [2]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[30] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[30]),
        .Q(\time_in_xfifo/fifo/src_in [30]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[31] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[31]),
        .Q(\time_in_xfifo/fifo/src_in [31]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[32] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[32]),
        .Q(\time_in_xfifo/fifo/src_in [32]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[33] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[33]),
        .Q(\time_in_xfifo/fifo/src_in [33]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[34] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[34]),
        .Q(\time_in_xfifo/fifo/src_in [34]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[35] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[35]),
        .Q(\time_in_xfifo/fifo/src_in [35]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[36] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[36]),
        .Q(\time_in_xfifo/fifo/src_in [36]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[37] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[37]),
        .Q(\time_in_xfifo/fifo/src_in [37]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[38] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[38]),
        .Q(\time_in_xfifo/fifo/src_in [38]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[39] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[39]),
        .Q(\time_in_xfifo/fifo/src_in [39]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[3] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[3]),
        .Q(\time_in_xfifo/fifo/src_in [3]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[40] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[40]),
        .Q(\time_in_xfifo/fifo/src_in [40]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[41] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[41]),
        .Q(\time_in_xfifo/fifo/src_in [41]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[42] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[42]),
        .Q(\time_in_xfifo/fifo/src_in [42]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[43] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[43]),
        .Q(\time_in_xfifo/fifo/src_in [43]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[44] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[44]),
        .Q(\time_in_xfifo/fifo/src_in [44]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[45] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[45]),
        .Q(\time_in_xfifo/fifo/src_in [45]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[46] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[46]),
        .Q(\time_in_xfifo/fifo/src_in [46]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[47] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[47]),
        .Q(\time_in_xfifo/fifo/src_in [47]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[48] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[48]),
        .Q(\time_in_xfifo/fifo/src_in [48]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[49] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[49]),
        .Q(\time_in_xfifo/fifo/src_in [49]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[4] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[4]),
        .Q(\time_in_xfifo/fifo/src_in [4]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[50] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[50]),
        .Q(\time_in_xfifo/fifo/src_in [50]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[51] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[51]),
        .Q(\time_in_xfifo/fifo/src_in [51]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[52] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[52]),
        .Q(\time_in_xfifo/fifo/src_in [52]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[53] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[53]),
        .Q(\time_in_xfifo/fifo/src_in [53]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[54] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[54]),
        .Q(\time_in_xfifo/fifo/src_in [54]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[55] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[55]),
        .Q(\time_in_xfifo/fifo/src_in [55]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[56] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[56]),
        .Q(\time_in_xfifo/fifo/src_in [56]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[57] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[57]),
        .Q(\time_in_xfifo/fifo/src_in [57]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[58] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[58]),
        .Q(\time_in_xfifo/fifo/src_in [58]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[59] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[59]),
        .Q(\time_in_xfifo/fifo/src_in [59]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[5] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[5]),
        .Q(\time_in_xfifo/fifo/src_in [5]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[60] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[60]),
        .Q(\time_in_xfifo/fifo/src_in [60]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[61] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[61]),
        .Q(\time_in_xfifo/fifo/src_in [61]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[62] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[62]),
        .Q(\time_in_xfifo/fifo/src_in [62]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[63] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[63]),
        .Q(\time_in_xfifo/fifo/src_in [63]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[64] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_asap),
        .Q(\time_in_xfifo/fifo/src_in [64]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[6] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[6]),
        .Q(\time_in_xfifo/fifo/src_in [6]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[7] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[7]),
        .Q(\time_in_xfifo/fifo/src_in [7]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[8] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[8]),
        .Q(\time_in_xfifo/fifo/src_in [8]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_in_reg[9] 
       (.C(SYS_clk),
        .CE(\time_in_xfifo/fifo/src_send03_out ),
        .D(SYS_time_in[9]),
        .Q(\time_in_xfifo/fifo/src_in [9]),
        .R(SYS_rst));
  FDRE \time_in_xfifo/fifo/src_send_reg 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(src_send_i_1_n_0),
        .Q(\time_in_xfifo/fifo/src_send ),
        .R(1'b0));
  FDRE \time_out_last/handshake/fifo/dest_ack_reg 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(dest_ack_i_1__1_n_0),
        .Q(\time_out_last/handshake/fifo/dest_ack ),
        .R(1'b0));
  (* DEST_EXT_HSK = "1" *) 
  (* DEST_SYNC_FF = "2" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_SYNC_FF = "2" *) 
  (* VERSION = "0" *) 
  (* WIDTH = "65" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_handshake \time_out_last/handshake/fifo/handshake 
       (.dest_ack(\time_out_last/handshake/fifo/dest_ack ),
        .dest_clk(SYS_clk),
        .dest_out({\time_out_last/handshake/fifo/handshake_n_1 ,\time_out_last/out_tdata }),
        .dest_req(\time_out_last/handshake/fifo/dest_req ),
        .src_clk(DATA_clk),
        .src_in({1'b0,\time_out_last/handshake/fifo/src_in }),
        .src_rcv(\time_out_last/handshake/fifo/src_rcv ),
        .src_send(\time_out_last/handshake/fifo/src_send ));
  FDRE \time_out_last/handshake/fifo/src_in_reg[0] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[0]),
        .Q(\time_out_last/handshake/fifo/src_in [0]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[10] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[10]),
        .Q(\time_out_last/handshake/fifo/src_in [10]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[11] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[11]),
        .Q(\time_out_last/handshake/fifo/src_in [11]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[12] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[12]),
        .Q(\time_out_last/handshake/fifo/src_in [12]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[13] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[13]),
        .Q(\time_out_last/handshake/fifo/src_in [13]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[14] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[14]),
        .Q(\time_out_last/handshake/fifo/src_in [14]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[15] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[15]),
        .Q(\time_out_last/handshake/fifo/src_in [15]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[16] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[16]),
        .Q(\time_out_last/handshake/fifo/src_in [16]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[17] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[17]),
        .Q(\time_out_last/handshake/fifo/src_in [17]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[18] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[18]),
        .Q(\time_out_last/handshake/fifo/src_in [18]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[19] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[19]),
        .Q(\time_out_last/handshake/fifo/src_in [19]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[1] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[1]),
        .Q(\time_out_last/handshake/fifo/src_in [1]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[20] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[20]),
        .Q(\time_out_last/handshake/fifo/src_in [20]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[21] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[21]),
        .Q(\time_out_last/handshake/fifo/src_in [21]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[22] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[22]),
        .Q(\time_out_last/handshake/fifo/src_in [22]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[23] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[23]),
        .Q(\time_out_last/handshake/fifo/src_in [23]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[24] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[24]),
        .Q(\time_out_last/handshake/fifo/src_in [24]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[25] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[25]),
        .Q(\time_out_last/handshake/fifo/src_in [25]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[26] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[26]),
        .Q(\time_out_last/handshake/fifo/src_in [26]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[27] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[27]),
        .Q(\time_out_last/handshake/fifo/src_in [27]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[28] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[28]),
        .Q(\time_out_last/handshake/fifo/src_in [28]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[29] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[29]),
        .Q(\time_out_last/handshake/fifo/src_in [29]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[2] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[2]),
        .Q(\time_out_last/handshake/fifo/src_in [2]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[30] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[30]),
        .Q(\time_out_last/handshake/fifo/src_in [30]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[31] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[31]),
        .Q(\time_out_last/handshake/fifo/src_in [31]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[32] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[32]),
        .Q(\time_out_last/handshake/fifo/src_in [32]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[33] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[33]),
        .Q(\time_out_last/handshake/fifo/src_in [33]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[34] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[34]),
        .Q(\time_out_last/handshake/fifo/src_in [34]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[35] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[35]),
        .Q(\time_out_last/handshake/fifo/src_in [35]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[36] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[36]),
        .Q(\time_out_last/handshake/fifo/src_in [36]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[37] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[37]),
        .Q(\time_out_last/handshake/fifo/src_in [37]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[38] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[38]),
        .Q(\time_out_last/handshake/fifo/src_in [38]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[39] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[39]),
        .Q(\time_out_last/handshake/fifo/src_in [39]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[3] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[3]),
        .Q(\time_out_last/handshake/fifo/src_in [3]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[40] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[40]),
        .Q(\time_out_last/handshake/fifo/src_in [40]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[41] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[41]),
        .Q(\time_out_last/handshake/fifo/src_in [41]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[42] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[42]),
        .Q(\time_out_last/handshake/fifo/src_in [42]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[43] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[43]),
        .Q(\time_out_last/handshake/fifo/src_in [43]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[44] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[44]),
        .Q(\time_out_last/handshake/fifo/src_in [44]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[45] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[45]),
        .Q(\time_out_last/handshake/fifo/src_in [45]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[46] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[46]),
        .Q(\time_out_last/handshake/fifo/src_in [46]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[47] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[47]),
        .Q(\time_out_last/handshake/fifo/src_in [47]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[48] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[48]),
        .Q(\time_out_last/handshake/fifo/src_in [48]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[49] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[49]),
        .Q(\time_out_last/handshake/fifo/src_in [49]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[4] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[4]),
        .Q(\time_out_last/handshake/fifo/src_in [4]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[50] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[50]),
        .Q(\time_out_last/handshake/fifo/src_in [50]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[51] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[51]),
        .Q(\time_out_last/handshake/fifo/src_in [51]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[52] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[52]),
        .Q(\time_out_last/handshake/fifo/src_in [52]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[53] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[53]),
        .Q(\time_out_last/handshake/fifo/src_in [53]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[54] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[54]),
        .Q(\time_out_last/handshake/fifo/src_in [54]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[55] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[55]),
        .Q(\time_out_last/handshake/fifo/src_in [55]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[56] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[56]),
        .Q(\time_out_last/handshake/fifo/src_in [56]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[57] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[57]),
        .Q(\time_out_last/handshake/fifo/src_in [57]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[58] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[58]),
        .Q(\time_out_last/handshake/fifo/src_in [58]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[59] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[59]),
        .Q(\time_out_last/handshake/fifo/src_in [59]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[5] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[5]),
        .Q(\time_out_last/handshake/fifo/src_in [5]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[60] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[60]),
        .Q(\time_out_last/handshake/fifo/src_in [60]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[61] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[61]),
        .Q(\time_out_last/handshake/fifo/src_in [61]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[62] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[62]),
        .Q(\time_out_last/handshake/fifo/src_in [62]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[63] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[63]),
        .Q(\time_out_last/handshake/fifo/src_in [63]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[6] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[6]),
        .Q(\time_out_last/handshake/fifo/src_in [6]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[7] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[7]),
        .Q(\time_out_last/handshake/fifo/src_in [7]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[8] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[8]),
        .Q(\time_out_last/handshake/fifo/src_in [8]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_in_reg[9] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1__0_n_0 ),
        .D(DATA_time_last[9]),
        .Q(\time_out_last/handshake/fifo/src_in [9]),
        .R(DATA_rst));
  FDRE \time_out_last/handshake/fifo/src_send_reg 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(src_send_i_1__1_n_0),
        .Q(\time_out_last/handshake/fifo/src_send ),
        .R(DATA_rst));
  FDRE \time_out_last/out_time_reg[0] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [0]),
        .Q(\time_out_last/out_time_reg_n_0_[0] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[10] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [10]),
        .Q(\time_out_last/out_time_reg_n_0_[10] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[11] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [11]),
        .Q(\time_out_last/out_time_reg_n_0_[11] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[12] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [12]),
        .Q(\time_out_last/out_time_reg_n_0_[12] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[13] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [13]),
        .Q(\time_out_last/out_time_reg_n_0_[13] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[14] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [14]),
        .Q(\time_out_last/out_time_reg_n_0_[14] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[15] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [15]),
        .Q(\time_out_last/out_time_reg_n_0_[15] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[16] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [16]),
        .Q(\time_out_last/out_time_reg_n_0_[16] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[17] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [17]),
        .Q(\time_out_last/out_time_reg_n_0_[17] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[18] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [18]),
        .Q(\time_out_last/out_time_reg_n_0_[18] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[19] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [19]),
        .Q(\time_out_last/out_time_reg_n_0_[19] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[1] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [1]),
        .Q(\time_out_last/out_time_reg_n_0_[1] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[20] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [20]),
        .Q(\time_out_last/out_time_reg_n_0_[20] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[21] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [21]),
        .Q(\time_out_last/out_time_reg_n_0_[21] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[22] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [22]),
        .Q(\time_out_last/out_time_reg_n_0_[22] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[23] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [23]),
        .Q(\time_out_last/out_time_reg_n_0_[23] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[24] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [24]),
        .Q(\time_out_last/out_time_reg_n_0_[24] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[25] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [25]),
        .Q(\time_out_last/out_time_reg_n_0_[25] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[26] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [26]),
        .Q(\time_out_last/out_time_reg_n_0_[26] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[27] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [27]),
        .Q(\time_out_last/out_time_reg_n_0_[27] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[28] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [28]),
        .Q(\time_out_last/out_time_reg_n_0_[28] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[29] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [29]),
        .Q(\time_out_last/out_time_reg_n_0_[29] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[2] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [2]),
        .Q(\time_out_last/out_time_reg_n_0_[2] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[30] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [30]),
        .Q(\time_out_last/out_time_reg_n_0_[30] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[31] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [31]),
        .Q(\time_out_last/out_time_reg_n_0_[31] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[32] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [32]),
        .Q(\time_out_last/out_time_reg_n_0_[32] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[33] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [33]),
        .Q(\time_out_last/out_time_reg_n_0_[33] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[34] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [34]),
        .Q(\time_out_last/out_time_reg_n_0_[34] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[35] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [35]),
        .Q(\time_out_last/out_time_reg_n_0_[35] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[36] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [36]),
        .Q(\time_out_last/out_time_reg_n_0_[36] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[37] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [37]),
        .Q(\time_out_last/out_time_reg_n_0_[37] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[38] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [38]),
        .Q(\time_out_last/out_time_reg_n_0_[38] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[39] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [39]),
        .Q(\time_out_last/out_time_reg_n_0_[39] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[3] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [3]),
        .Q(\time_out_last/out_time_reg_n_0_[3] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[40] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [40]),
        .Q(\time_out_last/out_time_reg_n_0_[40] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[41] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [41]),
        .Q(\time_out_last/out_time_reg_n_0_[41] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[42] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [42]),
        .Q(\time_out_last/out_time_reg_n_0_[42] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[43] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [43]),
        .Q(\time_out_last/out_time_reg_n_0_[43] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[44] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [44]),
        .Q(\time_out_last/out_time_reg_n_0_[44] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[45] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [45]),
        .Q(\time_out_last/out_time_reg_n_0_[45] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[46] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [46]),
        .Q(\time_out_last/out_time_reg_n_0_[46] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[47] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [47]),
        .Q(\time_out_last/out_time_reg_n_0_[47] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[48] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [48]),
        .Q(\time_out_last/out_time_reg_n_0_[48] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[49] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [49]),
        .Q(\time_out_last/out_time_reg_n_0_[49] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[4] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [4]),
        .Q(\time_out_last/out_time_reg_n_0_[4] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[50] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [50]),
        .Q(\time_out_last/out_time_reg_n_0_[50] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[51] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [51]),
        .Q(\time_out_last/out_time_reg_n_0_[51] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[52] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [52]),
        .Q(\time_out_last/out_time_reg_n_0_[52] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[53] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [53]),
        .Q(\time_out_last/out_time_reg_n_0_[53] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[54] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [54]),
        .Q(\time_out_last/out_time_reg_n_0_[54] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[55] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [55]),
        .Q(\time_out_last/out_time_reg_n_0_[55] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[56] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [56]),
        .Q(\time_out_last/out_time_reg_n_0_[56] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[57] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [57]),
        .Q(\time_out_last/out_time_reg_n_0_[57] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[58] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [58]),
        .Q(\time_out_last/out_time_reg_n_0_[58] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[59] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [59]),
        .Q(\time_out_last/out_time_reg_n_0_[59] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[5] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [5]),
        .Q(\time_out_last/out_time_reg_n_0_[5] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[60] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [60]),
        .Q(\time_out_last/out_time_reg_n_0_[60] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[61] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [61]),
        .Q(\time_out_last/out_time_reg_n_0_[61] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[62] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [62]),
        .Q(\time_out_last/out_time_reg_n_0_[62] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[63] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [63]),
        .Q(\time_out_last/out_time_reg_n_0_[63] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[6] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [6]),
        .Q(\time_out_last/out_time_reg_n_0_[6] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[7] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [7]),
        .Q(\time_out_last/out_time_reg_n_0_[7] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[8] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [8]),
        .Q(\time_out_last/out_time_reg_n_0_[8] ),
        .R(SYS_rst));
  FDRE \time_out_last/out_time_reg[9] 
       (.C(SYS_clk),
        .CE(\time_out_last/out_tvalid ),
        .D(\time_out_last/out_tdata [9]),
        .Q(\time_out_last/out_time_reg_n_0_[9] ),
        .R(SYS_rst));
  FDRE \time_out_now/handshake/fifo/dest_ack_reg 
       (.C(SYS_clk),
        .CE(1'b1),
        .D(dest_ack_i_1__0_n_0),
        .Q(\time_out_now/handshake/fifo/dest_ack ),
        .R(1'b0));
  (* DEST_EXT_HSK = "1" *) 
  (* DEST_SYNC_FF = "2" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_SYNC_FF = "2" *) 
  (* VERSION = "0" *) 
  (* WIDTH = "65" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_handshake__xdcDup__2 \time_out_now/handshake/fifo/handshake 
       (.dest_ack(\time_out_now/handshake/fifo/dest_ack ),
        .dest_clk(SYS_clk),
        .dest_out({\time_out_now/handshake/fifo/handshake_n_1 ,\time_out_now/out_tdata }),
        .dest_req(\time_out_now/handshake/fifo/dest_req ),
        .src_clk(DATA_clk),
        .src_in({1'b0,\time_out_now/handshake/fifo/src_in }),
        .src_rcv(\time_out_now/handshake/fifo/src_rcv ),
        .src_send(\time_out_now/handshake/fifo/src_send ));
  FDRE \time_out_now/handshake/fifo/src_in_reg[0] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[0]),
        .Q(\time_out_now/handshake/fifo/src_in [0]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[10] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[10]),
        .Q(\time_out_now/handshake/fifo/src_in [10]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[11] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[11]),
        .Q(\time_out_now/handshake/fifo/src_in [11]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[12] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[12]),
        .Q(\time_out_now/handshake/fifo/src_in [12]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[13] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[13]),
        .Q(\time_out_now/handshake/fifo/src_in [13]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[14] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[14]),
        .Q(\time_out_now/handshake/fifo/src_in [14]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[15] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[15]),
        .Q(\time_out_now/handshake/fifo/src_in [15]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[16] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[16]),
        .Q(\time_out_now/handshake/fifo/src_in [16]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[17] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[17]),
        .Q(\time_out_now/handshake/fifo/src_in [17]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[18] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[18]),
        .Q(\time_out_now/handshake/fifo/src_in [18]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[19] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[19]),
        .Q(\time_out_now/handshake/fifo/src_in [19]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[1] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[1]),
        .Q(\time_out_now/handshake/fifo/src_in [1]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[20] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[20]),
        .Q(\time_out_now/handshake/fifo/src_in [20]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[21] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[21]),
        .Q(\time_out_now/handshake/fifo/src_in [21]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[22] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[22]),
        .Q(\time_out_now/handshake/fifo/src_in [22]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[23] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[23]),
        .Q(\time_out_now/handshake/fifo/src_in [23]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[24] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[24]),
        .Q(\time_out_now/handshake/fifo/src_in [24]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[25] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[25]),
        .Q(\time_out_now/handshake/fifo/src_in [25]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[26] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[26]),
        .Q(\time_out_now/handshake/fifo/src_in [26]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[27] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[27]),
        .Q(\time_out_now/handshake/fifo/src_in [27]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[28] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[28]),
        .Q(\time_out_now/handshake/fifo/src_in [28]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[29] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[29]),
        .Q(\time_out_now/handshake/fifo/src_in [29]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[2] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[2]),
        .Q(\time_out_now/handshake/fifo/src_in [2]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[30] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[30]),
        .Q(\time_out_now/handshake/fifo/src_in [30]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[31] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[31]),
        .Q(\time_out_now/handshake/fifo/src_in [31]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[32] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[32]),
        .Q(\time_out_now/handshake/fifo/src_in [32]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[33] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[33]),
        .Q(\time_out_now/handshake/fifo/src_in [33]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[34] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[34]),
        .Q(\time_out_now/handshake/fifo/src_in [34]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[35] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[35]),
        .Q(\time_out_now/handshake/fifo/src_in [35]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[36] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[36]),
        .Q(\time_out_now/handshake/fifo/src_in [36]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[37] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[37]),
        .Q(\time_out_now/handshake/fifo/src_in [37]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[38] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[38]),
        .Q(\time_out_now/handshake/fifo/src_in [38]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[39] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[39]),
        .Q(\time_out_now/handshake/fifo/src_in [39]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[3] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[3]),
        .Q(\time_out_now/handshake/fifo/src_in [3]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[40] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[40]),
        .Q(\time_out_now/handshake/fifo/src_in [40]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[41] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[41]),
        .Q(\time_out_now/handshake/fifo/src_in [41]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[42] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[42]),
        .Q(\time_out_now/handshake/fifo/src_in [42]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[43] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[43]),
        .Q(\time_out_now/handshake/fifo/src_in [43]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[44] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[44]),
        .Q(\time_out_now/handshake/fifo/src_in [44]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[45] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[45]),
        .Q(\time_out_now/handshake/fifo/src_in [45]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[46] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[46]),
        .Q(\time_out_now/handshake/fifo/src_in [46]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[47] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[47]),
        .Q(\time_out_now/handshake/fifo/src_in [47]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[48] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[48]),
        .Q(\time_out_now/handshake/fifo/src_in [48]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[49] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[49]),
        .Q(\time_out_now/handshake/fifo/src_in [49]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[4] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[4]),
        .Q(\time_out_now/handshake/fifo/src_in [4]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[50] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[50]),
        .Q(\time_out_now/handshake/fifo/src_in [50]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[51] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[51]),
        .Q(\time_out_now/handshake/fifo/src_in [51]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[52] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[52]),
        .Q(\time_out_now/handshake/fifo/src_in [52]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[53] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[53]),
        .Q(\time_out_now/handshake/fifo/src_in [53]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[54] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[54]),
        .Q(\time_out_now/handshake/fifo/src_in [54]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[55] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[55]),
        .Q(\time_out_now/handshake/fifo/src_in [55]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[56] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[56]),
        .Q(\time_out_now/handshake/fifo/src_in [56]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[57] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[57]),
        .Q(\time_out_now/handshake/fifo/src_in [57]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[58] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[58]),
        .Q(\time_out_now/handshake/fifo/src_in [58]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[59] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[59]),
        .Q(\time_out_now/handshake/fifo/src_in [59]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[5] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[5]),
        .Q(\time_out_now/handshake/fifo/src_in [5]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[60] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[60]),
        .Q(\time_out_now/handshake/fifo/src_in [60]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[61] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[61]),
        .Q(\time_out_now/handshake/fifo/src_in [61]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[62] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[62]),
        .Q(\time_out_now/handshake/fifo/src_in [62]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[63] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[63]),
        .Q(\time_out_now/handshake/fifo/src_in [63]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[6] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[6]),
        .Q(\time_out_now/handshake/fifo/src_in [6]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[7] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[7]),
        .Q(\time_out_now/handshake/fifo/src_in [7]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[8] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[8]),
        .Q(\time_out_now/handshake/fifo/src_in [8]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_in_reg[9] 
       (.C(DATA_clk),
        .CE(\src_in[63]_i_1_n_0 ),
        .D(DATA_time_now[9]),
        .Q(\time_out_now/handshake/fifo/src_in [9]),
        .R(DATA_rst));
  FDRE \time_out_now/handshake/fifo/src_send_reg 
       (.C(DATA_clk),
        .CE(1'b1),
        .D(src_send_i_1__0_n_0),
        .Q(\time_out_now/handshake/fifo/src_send ),
        .R(DATA_rst));
  FDRE \time_out_now/out_time_reg[0] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [0]),
        .Q(out_time[0]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[10] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [10]),
        .Q(out_time[10]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[11] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [11]),
        .Q(out_time[11]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[12] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [12]),
        .Q(out_time[12]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[13] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [13]),
        .Q(out_time[13]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[14] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [14]),
        .Q(out_time[14]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[15] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [15]),
        .Q(out_time[15]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[16] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [16]),
        .Q(out_time[16]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[17] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [17]),
        .Q(out_time[17]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[18] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [18]),
        .Q(out_time[18]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[19] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [19]),
        .Q(out_time[19]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[1] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [1]),
        .Q(out_time[1]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[20] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [20]),
        .Q(out_time[20]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[21] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [21]),
        .Q(out_time[21]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[22] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [22]),
        .Q(out_time[22]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[23] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [23]),
        .Q(out_time[23]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[24] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [24]),
        .Q(out_time[24]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[25] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [25]),
        .Q(out_time[25]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[26] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [26]),
        .Q(out_time[26]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[27] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [27]),
        .Q(out_time[27]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[28] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [28]),
        .Q(out_time[28]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[29] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [29]),
        .Q(out_time[29]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[2] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [2]),
        .Q(out_time[2]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[30] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [30]),
        .Q(out_time[30]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[31] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [31]),
        .Q(out_time[31]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[32] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [32]),
        .Q(out_time[32]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[33] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [33]),
        .Q(out_time[33]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[34] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [34]),
        .Q(out_time[34]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[35] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [35]),
        .Q(out_time[35]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[36] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [36]),
        .Q(out_time[36]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[37] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [37]),
        .Q(out_time[37]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[38] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [38]),
        .Q(out_time[38]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[39] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [39]),
        .Q(out_time[39]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[3] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [3]),
        .Q(out_time[3]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[40] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [40]),
        .Q(out_time[40]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[41] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [41]),
        .Q(out_time[41]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[42] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [42]),
        .Q(out_time[42]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[43] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [43]),
        .Q(out_time[43]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[44] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [44]),
        .Q(out_time[44]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[45] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [45]),
        .Q(out_time[45]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[46] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [46]),
        .Q(out_time[46]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[47] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [47]),
        .Q(out_time[47]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[48] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [48]),
        .Q(out_time[48]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[49] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [49]),
        .Q(out_time[49]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[4] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [4]),
        .Q(out_time[4]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[50] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [50]),
        .Q(out_time[50]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[51] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [51]),
        .Q(out_time[51]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[52] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [52]),
        .Q(out_time[52]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[53] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [53]),
        .Q(out_time[53]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[54] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [54]),
        .Q(out_time[54]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[55] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [55]),
        .Q(out_time[55]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[56] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [56]),
        .Q(out_time[56]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[57] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [57]),
        .Q(out_time[57]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[58] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [58]),
        .Q(out_time[58]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[59] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [59]),
        .Q(out_time[59]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[5] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [5]),
        .Q(out_time[5]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[60] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [60]),
        .Q(out_time[60]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[61] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [61]),
        .Q(out_time[61]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[62] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [62]),
        .Q(out_time[62]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[63] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [63]),
        .Q(out_time[63]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[6] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [6]),
        .Q(out_time[6]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[7] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [7]),
        .Q(out_time[7]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[8] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [8]),
        .Q(out_time[8]),
        .R(SYS_rst));
  FDRE \time_out_now/out_time_reg[9] 
       (.C(SYS_clk),
        .CE(\time_out_now/out_tvalid ),
        .D(\time_out_now/out_tdata [9]),
        .Q(out_time[9]),
        .R(SYS_rst));
endmodule

(* DEST_EXT_HSK = "1" *) (* DEST_SYNC_FF = "2" *) (* INIT_SYNC_FF = "0" *) 
(* SIM_ASSERT_CHK = "0" *) (* SRC_SYNC_FF = "2" *) (* VERSION = "0" *) 
(* WIDTH = "65" *) (* XPM_MODULE = "TRUE" *) (* xpm_cdc = "HANDSHAKE" *) 
module xpm_cdc_handshake
   (src_clk,
    src_in,
    src_send,
    src_rcv,
    dest_clk,
    dest_out,
    dest_req,
    dest_ack);
  input src_clk;
  input [64:0]src_in;
  input src_send;
  output src_rcv;
  input dest_clk;
  output [64:0]dest_out;
  output dest_req;
  input dest_ack;

  wire dest_ack;
  wire dest_clk;
  (* DIRECT_ENABLE *) wire dest_hsdata_en;
  (* RTL_KEEP = "true" *) (* xpm_cdc = "HANDSHAKE" *) wire [64:0]dest_hsdata_ff;
  wire dest_req;
  wire dest_req_nxt;
  wire p_0_in;
  wire src_clk;
  wire [64:0]src_hsdata_ff;
  wire [64:0]src_in;
  wire src_rcv;
  wire src_send;
  wire src_sendd_ff;

  assign dest_out[64:0] = dest_hsdata_ff;
  LUT2 #(
    .INIT(4'h2)) 
    dest_hsdata_en_inferred_i_1
       (.I0(dest_req_nxt),
        .I1(dest_req),
        .O(dest_hsdata_en));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[0] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[0]),
        .Q(dest_hsdata_ff[0]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[10] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[10]),
        .Q(dest_hsdata_ff[10]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[11] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[11]),
        .Q(dest_hsdata_ff[11]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[12] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[12]),
        .Q(dest_hsdata_ff[12]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[13] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[13]),
        .Q(dest_hsdata_ff[13]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[14] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[14]),
        .Q(dest_hsdata_ff[14]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[15] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[15]),
        .Q(dest_hsdata_ff[15]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[16] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[16]),
        .Q(dest_hsdata_ff[16]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[17] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[17]),
        .Q(dest_hsdata_ff[17]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[18] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[18]),
        .Q(dest_hsdata_ff[18]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[19] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[19]),
        .Q(dest_hsdata_ff[19]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[1] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[1]),
        .Q(dest_hsdata_ff[1]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[20] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[20]),
        .Q(dest_hsdata_ff[20]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[21] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[21]),
        .Q(dest_hsdata_ff[21]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[22] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[22]),
        .Q(dest_hsdata_ff[22]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[23] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[23]),
        .Q(dest_hsdata_ff[23]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[24] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[24]),
        .Q(dest_hsdata_ff[24]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[25] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[25]),
        .Q(dest_hsdata_ff[25]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[26] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[26]),
        .Q(dest_hsdata_ff[26]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[27] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[27]),
        .Q(dest_hsdata_ff[27]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[28] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[28]),
        .Q(dest_hsdata_ff[28]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[29] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[29]),
        .Q(dest_hsdata_ff[29]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[2] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[2]),
        .Q(dest_hsdata_ff[2]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[30] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[30]),
        .Q(dest_hsdata_ff[30]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[31] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[31]),
        .Q(dest_hsdata_ff[31]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[32] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[32]),
        .Q(dest_hsdata_ff[32]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[33] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[33]),
        .Q(dest_hsdata_ff[33]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[34] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[34]),
        .Q(dest_hsdata_ff[34]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[35] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[35]),
        .Q(dest_hsdata_ff[35]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[36] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[36]),
        .Q(dest_hsdata_ff[36]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[37] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[37]),
        .Q(dest_hsdata_ff[37]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[38] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[38]),
        .Q(dest_hsdata_ff[38]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[39] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[39]),
        .Q(dest_hsdata_ff[39]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[3] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[3]),
        .Q(dest_hsdata_ff[3]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[40] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[40]),
        .Q(dest_hsdata_ff[40]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[41] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[41]),
        .Q(dest_hsdata_ff[41]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[42] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[42]),
        .Q(dest_hsdata_ff[42]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[43] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[43]),
        .Q(dest_hsdata_ff[43]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[44] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[44]),
        .Q(dest_hsdata_ff[44]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[45] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[45]),
        .Q(dest_hsdata_ff[45]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[46] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[46]),
        .Q(dest_hsdata_ff[46]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[47] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[47]),
        .Q(dest_hsdata_ff[47]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[48] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[48]),
        .Q(dest_hsdata_ff[48]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[49] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[49]),
        .Q(dest_hsdata_ff[49]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[4] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[4]),
        .Q(dest_hsdata_ff[4]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[50] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[50]),
        .Q(dest_hsdata_ff[50]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[51] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[51]),
        .Q(dest_hsdata_ff[51]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[52] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[52]),
        .Q(dest_hsdata_ff[52]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[53] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[53]),
        .Q(dest_hsdata_ff[53]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[54] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[54]),
        .Q(dest_hsdata_ff[54]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[55] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[55]),
        .Q(dest_hsdata_ff[55]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[56] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[56]),
        .Q(dest_hsdata_ff[56]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[57] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[57]),
        .Q(dest_hsdata_ff[57]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[58] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[58]),
        .Q(dest_hsdata_ff[58]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[59] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[59]),
        .Q(dest_hsdata_ff[59]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[5] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[5]),
        .Q(dest_hsdata_ff[5]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[60] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[60]),
        .Q(dest_hsdata_ff[60]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[61] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[61]),
        .Q(dest_hsdata_ff[61]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[62] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[62]),
        .Q(dest_hsdata_ff[62]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[63] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[63]),
        .Q(dest_hsdata_ff[63]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[64] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[64]),
        .Q(dest_hsdata_ff[64]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[6] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[6]),
        .Q(dest_hsdata_ff[6]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[7] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[7]),
        .Q(dest_hsdata_ff[7]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[8] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[8]),
        .Q(dest_hsdata_ff[8]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[9] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[9]),
        .Q(dest_hsdata_ff[9]),
        .R(1'b0));
  FDRE dest_req_ff_reg
       (.C(dest_clk),
        .CE(1'b1),
        .D(dest_req_nxt),
        .Q(dest_req),
        .R(1'b0));
  LUT1 #(
    .INIT(2'h1)) 
    \src_hsdata_ff[64]_i_1 
       (.I0(src_sendd_ff),
        .O(p_0_in));
  FDRE \src_hsdata_ff_reg[0] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[0]),
        .Q(src_hsdata_ff[0]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[10] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[10]),
        .Q(src_hsdata_ff[10]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[11] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[11]),
        .Q(src_hsdata_ff[11]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[12] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[12]),
        .Q(src_hsdata_ff[12]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[13] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[13]),
        .Q(src_hsdata_ff[13]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[14] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[14]),
        .Q(src_hsdata_ff[14]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[15] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[15]),
        .Q(src_hsdata_ff[15]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[16] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[16]),
        .Q(src_hsdata_ff[16]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[17] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[17]),
        .Q(src_hsdata_ff[17]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[18] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[18]),
        .Q(src_hsdata_ff[18]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[19] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[19]),
        .Q(src_hsdata_ff[19]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[1] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[1]),
        .Q(src_hsdata_ff[1]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[20] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[20]),
        .Q(src_hsdata_ff[20]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[21] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[21]),
        .Q(src_hsdata_ff[21]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[22] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[22]),
        .Q(src_hsdata_ff[22]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[23] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[23]),
        .Q(src_hsdata_ff[23]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[24] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[24]),
        .Q(src_hsdata_ff[24]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[25] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[25]),
        .Q(src_hsdata_ff[25]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[26] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[26]),
        .Q(src_hsdata_ff[26]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[27] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[27]),
        .Q(src_hsdata_ff[27]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[28] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[28]),
        .Q(src_hsdata_ff[28]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[29] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[29]),
        .Q(src_hsdata_ff[29]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[2] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[2]),
        .Q(src_hsdata_ff[2]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[30] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[30]),
        .Q(src_hsdata_ff[30]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[31] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[31]),
        .Q(src_hsdata_ff[31]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[32] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[32]),
        .Q(src_hsdata_ff[32]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[33] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[33]),
        .Q(src_hsdata_ff[33]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[34] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[34]),
        .Q(src_hsdata_ff[34]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[35] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[35]),
        .Q(src_hsdata_ff[35]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[36] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[36]),
        .Q(src_hsdata_ff[36]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[37] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[37]),
        .Q(src_hsdata_ff[37]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[38] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[38]),
        .Q(src_hsdata_ff[38]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[39] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[39]),
        .Q(src_hsdata_ff[39]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[3] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[3]),
        .Q(src_hsdata_ff[3]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[40] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[40]),
        .Q(src_hsdata_ff[40]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[41] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[41]),
        .Q(src_hsdata_ff[41]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[42] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[42]),
        .Q(src_hsdata_ff[42]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[43] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[43]),
        .Q(src_hsdata_ff[43]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[44] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[44]),
        .Q(src_hsdata_ff[44]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[45] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[45]),
        .Q(src_hsdata_ff[45]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[46] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[46]),
        .Q(src_hsdata_ff[46]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[47] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[47]),
        .Q(src_hsdata_ff[47]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[48] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[48]),
        .Q(src_hsdata_ff[48]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[49] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[49]),
        .Q(src_hsdata_ff[49]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[4] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[4]),
        .Q(src_hsdata_ff[4]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[50] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[50]),
        .Q(src_hsdata_ff[50]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[51] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[51]),
        .Q(src_hsdata_ff[51]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[52] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[52]),
        .Q(src_hsdata_ff[52]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[53] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[53]),
        .Q(src_hsdata_ff[53]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[54] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[54]),
        .Q(src_hsdata_ff[54]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[55] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[55]),
        .Q(src_hsdata_ff[55]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[56] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[56]),
        .Q(src_hsdata_ff[56]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[57] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[57]),
        .Q(src_hsdata_ff[57]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[58] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[58]),
        .Q(src_hsdata_ff[58]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[59] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[59]),
        .Q(src_hsdata_ff[59]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[5] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[5]),
        .Q(src_hsdata_ff[5]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[60] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[60]),
        .Q(src_hsdata_ff[60]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[61] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[61]),
        .Q(src_hsdata_ff[61]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[62] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[62]),
        .Q(src_hsdata_ff[62]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[63] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[63]),
        .Q(src_hsdata_ff[63]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[64] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[64]),
        .Q(src_hsdata_ff[64]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[6] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[6]),
        .Q(src_hsdata_ff[6]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[7] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[7]),
        .Q(src_hsdata_ff[7]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[8] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[8]),
        .Q(src_hsdata_ff[8]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[9] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[9]),
        .Q(src_hsdata_ff[9]),
        .R(1'b0));
  FDRE src_sendd_ff_reg
       (.C(src_clk),
        .CE(1'b1),
        .D(src_send),
        .Q(src_sendd_ff),
        .R(1'b0));
  (* DEST_SYNC_FF = "2" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_INPUT_REG = "0" *) 
  (* VERSION = "0" *) 
  (* XPM_CDC = "SINGLE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_single xpm_cdc_single_dest2src_inst
       (.dest_clk(src_clk),
        .dest_out(src_rcv),
        .src_clk(dest_clk),
        .src_in(dest_ack));
  (* DEST_SYNC_FF = "2" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_INPUT_REG = "0" *) 
  (* VERSION = "0" *) 
  (* XPM_CDC = "SINGLE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_single__10 xpm_cdc_single_src2dest_inst
       (.dest_clk(dest_clk),
        .dest_out(dest_req_nxt),
        .src_clk(src_clk),
        .src_in(src_sendd_ff));
endmodule

(* DEST_EXT_HSK = "1" *) (* DEST_SYNC_FF = "2" *) (* INIT_SYNC_FF = "0" *) 
(* ORIG_REF_NAME = "xpm_cdc_handshake" *) (* SIM_ASSERT_CHK = "0" *) (* SRC_SYNC_FF = "2" *) 
(* VERSION = "0" *) (* WIDTH = "65" *) (* XPM_MODULE = "TRUE" *) 
(* xpm_cdc = "HANDSHAKE" *) 
module xpm_cdc_handshake__xdcDup__1
   (src_clk,
    src_in,
    src_send,
    src_rcv,
    dest_clk,
    dest_out,
    dest_req,
    dest_ack);
  input src_clk;
  input [64:0]src_in;
  input src_send;
  output src_rcv;
  input dest_clk;
  output [64:0]dest_out;
  output dest_req;
  input dest_ack;

  wire dest_ack;
  wire dest_clk;
  (* DIRECT_ENABLE *) wire dest_hsdata_en;
  (* RTL_KEEP = "true" *) (* xpm_cdc = "HANDSHAKE" *) wire [64:0]dest_hsdata_ff;
  wire dest_req;
  wire dest_req_nxt;
  wire p_0_in;
  wire src_clk;
  wire [64:0]src_hsdata_ff;
  wire [64:0]src_in;
  wire src_rcv;
  wire src_send;
  wire src_sendd_ff;

  assign dest_out[64:0] = dest_hsdata_ff;
  LUT2 #(
    .INIT(4'h2)) 
    dest_hsdata_en_inferred_i_1
       (.I0(dest_req_nxt),
        .I1(dest_req),
        .O(dest_hsdata_en));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[0] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[0]),
        .Q(dest_hsdata_ff[0]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[10] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[10]),
        .Q(dest_hsdata_ff[10]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[11] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[11]),
        .Q(dest_hsdata_ff[11]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[12] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[12]),
        .Q(dest_hsdata_ff[12]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[13] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[13]),
        .Q(dest_hsdata_ff[13]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[14] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[14]),
        .Q(dest_hsdata_ff[14]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[15] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[15]),
        .Q(dest_hsdata_ff[15]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[16] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[16]),
        .Q(dest_hsdata_ff[16]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[17] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[17]),
        .Q(dest_hsdata_ff[17]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[18] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[18]),
        .Q(dest_hsdata_ff[18]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[19] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[19]),
        .Q(dest_hsdata_ff[19]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[1] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[1]),
        .Q(dest_hsdata_ff[1]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[20] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[20]),
        .Q(dest_hsdata_ff[20]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[21] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[21]),
        .Q(dest_hsdata_ff[21]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[22] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[22]),
        .Q(dest_hsdata_ff[22]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[23] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[23]),
        .Q(dest_hsdata_ff[23]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[24] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[24]),
        .Q(dest_hsdata_ff[24]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[25] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[25]),
        .Q(dest_hsdata_ff[25]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[26] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[26]),
        .Q(dest_hsdata_ff[26]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[27] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[27]),
        .Q(dest_hsdata_ff[27]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[28] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[28]),
        .Q(dest_hsdata_ff[28]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[29] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[29]),
        .Q(dest_hsdata_ff[29]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[2] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[2]),
        .Q(dest_hsdata_ff[2]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[30] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[30]),
        .Q(dest_hsdata_ff[30]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[31] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[31]),
        .Q(dest_hsdata_ff[31]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[32] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[32]),
        .Q(dest_hsdata_ff[32]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[33] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[33]),
        .Q(dest_hsdata_ff[33]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[34] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[34]),
        .Q(dest_hsdata_ff[34]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[35] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[35]),
        .Q(dest_hsdata_ff[35]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[36] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[36]),
        .Q(dest_hsdata_ff[36]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[37] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[37]),
        .Q(dest_hsdata_ff[37]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[38] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[38]),
        .Q(dest_hsdata_ff[38]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[39] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[39]),
        .Q(dest_hsdata_ff[39]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[3] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[3]),
        .Q(dest_hsdata_ff[3]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[40] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[40]),
        .Q(dest_hsdata_ff[40]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[41] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[41]),
        .Q(dest_hsdata_ff[41]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[42] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[42]),
        .Q(dest_hsdata_ff[42]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[43] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[43]),
        .Q(dest_hsdata_ff[43]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[44] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[44]),
        .Q(dest_hsdata_ff[44]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[45] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[45]),
        .Q(dest_hsdata_ff[45]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[46] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[46]),
        .Q(dest_hsdata_ff[46]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[47] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[47]),
        .Q(dest_hsdata_ff[47]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[48] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[48]),
        .Q(dest_hsdata_ff[48]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[49] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[49]),
        .Q(dest_hsdata_ff[49]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[4] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[4]),
        .Q(dest_hsdata_ff[4]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[50] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[50]),
        .Q(dest_hsdata_ff[50]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[51] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[51]),
        .Q(dest_hsdata_ff[51]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[52] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[52]),
        .Q(dest_hsdata_ff[52]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[53] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[53]),
        .Q(dest_hsdata_ff[53]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[54] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[54]),
        .Q(dest_hsdata_ff[54]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[55] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[55]),
        .Q(dest_hsdata_ff[55]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[56] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[56]),
        .Q(dest_hsdata_ff[56]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[57] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[57]),
        .Q(dest_hsdata_ff[57]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[58] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[58]),
        .Q(dest_hsdata_ff[58]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[59] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[59]),
        .Q(dest_hsdata_ff[59]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[5] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[5]),
        .Q(dest_hsdata_ff[5]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[60] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[60]),
        .Q(dest_hsdata_ff[60]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[61] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[61]),
        .Q(dest_hsdata_ff[61]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[62] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[62]),
        .Q(dest_hsdata_ff[62]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[63] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[63]),
        .Q(dest_hsdata_ff[63]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[64] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[64]),
        .Q(dest_hsdata_ff[64]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[6] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[6]),
        .Q(dest_hsdata_ff[6]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[7] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[7]),
        .Q(dest_hsdata_ff[7]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[8] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[8]),
        .Q(dest_hsdata_ff[8]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[9] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[9]),
        .Q(dest_hsdata_ff[9]),
        .R(1'b0));
  FDRE dest_req_ff_reg
       (.C(dest_clk),
        .CE(1'b1),
        .D(dest_req_nxt),
        .Q(dest_req),
        .R(1'b0));
  LUT1 #(
    .INIT(2'h1)) 
    \src_hsdata_ff[64]_i_1 
       (.I0(src_sendd_ff),
        .O(p_0_in));
  FDRE \src_hsdata_ff_reg[0] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[0]),
        .Q(src_hsdata_ff[0]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[10] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[10]),
        .Q(src_hsdata_ff[10]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[11] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[11]),
        .Q(src_hsdata_ff[11]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[12] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[12]),
        .Q(src_hsdata_ff[12]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[13] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[13]),
        .Q(src_hsdata_ff[13]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[14] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[14]),
        .Q(src_hsdata_ff[14]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[15] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[15]),
        .Q(src_hsdata_ff[15]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[16] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[16]),
        .Q(src_hsdata_ff[16]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[17] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[17]),
        .Q(src_hsdata_ff[17]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[18] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[18]),
        .Q(src_hsdata_ff[18]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[19] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[19]),
        .Q(src_hsdata_ff[19]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[1] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[1]),
        .Q(src_hsdata_ff[1]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[20] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[20]),
        .Q(src_hsdata_ff[20]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[21] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[21]),
        .Q(src_hsdata_ff[21]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[22] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[22]),
        .Q(src_hsdata_ff[22]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[23] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[23]),
        .Q(src_hsdata_ff[23]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[24] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[24]),
        .Q(src_hsdata_ff[24]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[25] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[25]),
        .Q(src_hsdata_ff[25]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[26] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[26]),
        .Q(src_hsdata_ff[26]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[27] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[27]),
        .Q(src_hsdata_ff[27]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[28] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[28]),
        .Q(src_hsdata_ff[28]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[29] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[29]),
        .Q(src_hsdata_ff[29]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[2] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[2]),
        .Q(src_hsdata_ff[2]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[30] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[30]),
        .Q(src_hsdata_ff[30]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[31] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[31]),
        .Q(src_hsdata_ff[31]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[32] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[32]),
        .Q(src_hsdata_ff[32]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[33] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[33]),
        .Q(src_hsdata_ff[33]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[34] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[34]),
        .Q(src_hsdata_ff[34]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[35] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[35]),
        .Q(src_hsdata_ff[35]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[36] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[36]),
        .Q(src_hsdata_ff[36]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[37] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[37]),
        .Q(src_hsdata_ff[37]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[38] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[38]),
        .Q(src_hsdata_ff[38]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[39] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[39]),
        .Q(src_hsdata_ff[39]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[3] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[3]),
        .Q(src_hsdata_ff[3]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[40] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[40]),
        .Q(src_hsdata_ff[40]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[41] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[41]),
        .Q(src_hsdata_ff[41]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[42] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[42]),
        .Q(src_hsdata_ff[42]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[43] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[43]),
        .Q(src_hsdata_ff[43]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[44] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[44]),
        .Q(src_hsdata_ff[44]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[45] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[45]),
        .Q(src_hsdata_ff[45]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[46] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[46]),
        .Q(src_hsdata_ff[46]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[47] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[47]),
        .Q(src_hsdata_ff[47]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[48] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[48]),
        .Q(src_hsdata_ff[48]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[49] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[49]),
        .Q(src_hsdata_ff[49]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[4] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[4]),
        .Q(src_hsdata_ff[4]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[50] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[50]),
        .Q(src_hsdata_ff[50]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[51] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[51]),
        .Q(src_hsdata_ff[51]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[52] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[52]),
        .Q(src_hsdata_ff[52]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[53] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[53]),
        .Q(src_hsdata_ff[53]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[54] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[54]),
        .Q(src_hsdata_ff[54]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[55] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[55]),
        .Q(src_hsdata_ff[55]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[56] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[56]),
        .Q(src_hsdata_ff[56]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[57] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[57]),
        .Q(src_hsdata_ff[57]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[58] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[58]),
        .Q(src_hsdata_ff[58]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[59] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[59]),
        .Q(src_hsdata_ff[59]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[5] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[5]),
        .Q(src_hsdata_ff[5]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[60] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[60]),
        .Q(src_hsdata_ff[60]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[61] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[61]),
        .Q(src_hsdata_ff[61]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[62] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[62]),
        .Q(src_hsdata_ff[62]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[63] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[63]),
        .Q(src_hsdata_ff[63]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[64] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[64]),
        .Q(src_hsdata_ff[64]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[6] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[6]),
        .Q(src_hsdata_ff[6]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[7] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[7]),
        .Q(src_hsdata_ff[7]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[8] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[8]),
        .Q(src_hsdata_ff[8]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[9] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[9]),
        .Q(src_hsdata_ff[9]),
        .R(1'b0));
  FDRE src_sendd_ff_reg
       (.C(src_clk),
        .CE(1'b1),
        .D(src_send),
        .Q(src_sendd_ff),
        .R(1'b0));
  (* DEST_SYNC_FF = "2" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_INPUT_REG = "0" *) 
  (* VERSION = "0" *) 
  (* XPM_CDC = "SINGLE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_single__7 xpm_cdc_single_dest2src_inst
       (.dest_clk(src_clk),
        .dest_out(src_rcv),
        .src_clk(dest_clk),
        .src_in(dest_ack));
  (* DEST_SYNC_FF = "2" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_INPUT_REG = "0" *) 
  (* VERSION = "0" *) 
  (* XPM_CDC = "SINGLE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_single__6 xpm_cdc_single_src2dest_inst
       (.dest_clk(dest_clk),
        .dest_out(dest_req_nxt),
        .src_clk(src_clk),
        .src_in(src_sendd_ff));
endmodule

(* DEST_EXT_HSK = "1" *) (* DEST_SYNC_FF = "2" *) (* INIT_SYNC_FF = "0" *) 
(* ORIG_REF_NAME = "xpm_cdc_handshake" *) (* SIM_ASSERT_CHK = "0" *) (* SRC_SYNC_FF = "2" *) 
(* VERSION = "0" *) (* WIDTH = "65" *) (* XPM_MODULE = "TRUE" *) 
(* xpm_cdc = "HANDSHAKE" *) 
module xpm_cdc_handshake__xdcDup__2
   (src_clk,
    src_in,
    src_send,
    src_rcv,
    dest_clk,
    dest_out,
    dest_req,
    dest_ack);
  input src_clk;
  input [64:0]src_in;
  input src_send;
  output src_rcv;
  input dest_clk;
  output [64:0]dest_out;
  output dest_req;
  input dest_ack;

  wire dest_ack;
  wire dest_clk;
  (* DIRECT_ENABLE *) wire dest_hsdata_en;
  (* RTL_KEEP = "true" *) (* xpm_cdc = "HANDSHAKE" *) wire [64:0]dest_hsdata_ff;
  wire dest_req;
  wire dest_req_nxt;
  wire p_0_in;
  wire src_clk;
  wire [64:0]src_hsdata_ff;
  wire [64:0]src_in;
  wire src_rcv;
  wire src_send;
  wire src_sendd_ff;

  assign dest_out[64:0] = dest_hsdata_ff;
  LUT2 #(
    .INIT(4'h2)) 
    dest_hsdata_en_inferred_i_1
       (.I0(dest_req_nxt),
        .I1(dest_req),
        .O(dest_hsdata_en));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[0] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[0]),
        .Q(dest_hsdata_ff[0]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[10] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[10]),
        .Q(dest_hsdata_ff[10]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[11] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[11]),
        .Q(dest_hsdata_ff[11]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[12] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[12]),
        .Q(dest_hsdata_ff[12]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[13] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[13]),
        .Q(dest_hsdata_ff[13]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[14] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[14]),
        .Q(dest_hsdata_ff[14]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[15] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[15]),
        .Q(dest_hsdata_ff[15]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[16] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[16]),
        .Q(dest_hsdata_ff[16]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[17] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[17]),
        .Q(dest_hsdata_ff[17]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[18] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[18]),
        .Q(dest_hsdata_ff[18]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[19] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[19]),
        .Q(dest_hsdata_ff[19]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[1] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[1]),
        .Q(dest_hsdata_ff[1]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[20] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[20]),
        .Q(dest_hsdata_ff[20]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[21] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[21]),
        .Q(dest_hsdata_ff[21]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[22] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[22]),
        .Q(dest_hsdata_ff[22]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[23] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[23]),
        .Q(dest_hsdata_ff[23]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[24] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[24]),
        .Q(dest_hsdata_ff[24]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[25] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[25]),
        .Q(dest_hsdata_ff[25]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[26] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[26]),
        .Q(dest_hsdata_ff[26]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[27] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[27]),
        .Q(dest_hsdata_ff[27]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[28] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[28]),
        .Q(dest_hsdata_ff[28]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[29] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[29]),
        .Q(dest_hsdata_ff[29]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[2] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[2]),
        .Q(dest_hsdata_ff[2]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[30] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[30]),
        .Q(dest_hsdata_ff[30]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[31] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[31]),
        .Q(dest_hsdata_ff[31]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[32] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[32]),
        .Q(dest_hsdata_ff[32]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[33] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[33]),
        .Q(dest_hsdata_ff[33]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[34] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[34]),
        .Q(dest_hsdata_ff[34]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[35] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[35]),
        .Q(dest_hsdata_ff[35]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[36] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[36]),
        .Q(dest_hsdata_ff[36]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[37] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[37]),
        .Q(dest_hsdata_ff[37]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[38] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[38]),
        .Q(dest_hsdata_ff[38]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[39] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[39]),
        .Q(dest_hsdata_ff[39]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[3] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[3]),
        .Q(dest_hsdata_ff[3]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[40] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[40]),
        .Q(dest_hsdata_ff[40]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[41] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[41]),
        .Q(dest_hsdata_ff[41]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[42] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[42]),
        .Q(dest_hsdata_ff[42]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[43] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[43]),
        .Q(dest_hsdata_ff[43]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[44] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[44]),
        .Q(dest_hsdata_ff[44]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[45] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[45]),
        .Q(dest_hsdata_ff[45]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[46] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[46]),
        .Q(dest_hsdata_ff[46]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[47] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[47]),
        .Q(dest_hsdata_ff[47]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[48] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[48]),
        .Q(dest_hsdata_ff[48]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[49] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[49]),
        .Q(dest_hsdata_ff[49]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[4] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[4]),
        .Q(dest_hsdata_ff[4]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[50] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[50]),
        .Q(dest_hsdata_ff[50]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[51] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[51]),
        .Q(dest_hsdata_ff[51]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[52] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[52]),
        .Q(dest_hsdata_ff[52]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[53] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[53]),
        .Q(dest_hsdata_ff[53]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[54] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[54]),
        .Q(dest_hsdata_ff[54]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[55] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[55]),
        .Q(dest_hsdata_ff[55]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[56] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[56]),
        .Q(dest_hsdata_ff[56]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[57] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[57]),
        .Q(dest_hsdata_ff[57]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[58] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[58]),
        .Q(dest_hsdata_ff[58]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[59] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[59]),
        .Q(dest_hsdata_ff[59]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[5] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[5]),
        .Q(dest_hsdata_ff[5]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[60] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[60]),
        .Q(dest_hsdata_ff[60]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[61] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[61]),
        .Q(dest_hsdata_ff[61]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[62] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[62]),
        .Q(dest_hsdata_ff[62]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[63] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[63]),
        .Q(dest_hsdata_ff[63]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[64] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[64]),
        .Q(dest_hsdata_ff[64]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[6] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[6]),
        .Q(dest_hsdata_ff[6]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[7] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[7]),
        .Q(dest_hsdata_ff[7]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[8] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[8]),
        .Q(dest_hsdata_ff[8]),
        .R(1'b0));
  (* KEEP = "true" *) 
  (* XPM_CDC = "HANDSHAKE" *) 
  FDRE \dest_hsdata_ff_reg[9] 
       (.C(dest_clk),
        .CE(dest_hsdata_en),
        .D(src_hsdata_ff[9]),
        .Q(dest_hsdata_ff[9]),
        .R(1'b0));
  FDRE dest_req_ff_reg
       (.C(dest_clk),
        .CE(1'b1),
        .D(dest_req_nxt),
        .Q(dest_req),
        .R(1'b0));
  LUT1 #(
    .INIT(2'h1)) 
    \src_hsdata_ff[64]_i_1 
       (.I0(src_sendd_ff),
        .O(p_0_in));
  FDRE \src_hsdata_ff_reg[0] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[0]),
        .Q(src_hsdata_ff[0]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[10] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[10]),
        .Q(src_hsdata_ff[10]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[11] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[11]),
        .Q(src_hsdata_ff[11]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[12] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[12]),
        .Q(src_hsdata_ff[12]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[13] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[13]),
        .Q(src_hsdata_ff[13]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[14] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[14]),
        .Q(src_hsdata_ff[14]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[15] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[15]),
        .Q(src_hsdata_ff[15]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[16] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[16]),
        .Q(src_hsdata_ff[16]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[17] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[17]),
        .Q(src_hsdata_ff[17]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[18] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[18]),
        .Q(src_hsdata_ff[18]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[19] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[19]),
        .Q(src_hsdata_ff[19]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[1] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[1]),
        .Q(src_hsdata_ff[1]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[20] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[20]),
        .Q(src_hsdata_ff[20]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[21] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[21]),
        .Q(src_hsdata_ff[21]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[22] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[22]),
        .Q(src_hsdata_ff[22]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[23] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[23]),
        .Q(src_hsdata_ff[23]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[24] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[24]),
        .Q(src_hsdata_ff[24]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[25] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[25]),
        .Q(src_hsdata_ff[25]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[26] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[26]),
        .Q(src_hsdata_ff[26]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[27] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[27]),
        .Q(src_hsdata_ff[27]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[28] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[28]),
        .Q(src_hsdata_ff[28]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[29] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[29]),
        .Q(src_hsdata_ff[29]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[2] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[2]),
        .Q(src_hsdata_ff[2]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[30] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[30]),
        .Q(src_hsdata_ff[30]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[31] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[31]),
        .Q(src_hsdata_ff[31]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[32] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[32]),
        .Q(src_hsdata_ff[32]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[33] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[33]),
        .Q(src_hsdata_ff[33]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[34] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[34]),
        .Q(src_hsdata_ff[34]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[35] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[35]),
        .Q(src_hsdata_ff[35]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[36] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[36]),
        .Q(src_hsdata_ff[36]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[37] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[37]),
        .Q(src_hsdata_ff[37]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[38] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[38]),
        .Q(src_hsdata_ff[38]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[39] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[39]),
        .Q(src_hsdata_ff[39]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[3] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[3]),
        .Q(src_hsdata_ff[3]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[40] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[40]),
        .Q(src_hsdata_ff[40]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[41] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[41]),
        .Q(src_hsdata_ff[41]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[42] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[42]),
        .Q(src_hsdata_ff[42]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[43] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[43]),
        .Q(src_hsdata_ff[43]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[44] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[44]),
        .Q(src_hsdata_ff[44]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[45] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[45]),
        .Q(src_hsdata_ff[45]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[46] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[46]),
        .Q(src_hsdata_ff[46]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[47] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[47]),
        .Q(src_hsdata_ff[47]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[48] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[48]),
        .Q(src_hsdata_ff[48]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[49] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[49]),
        .Q(src_hsdata_ff[49]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[4] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[4]),
        .Q(src_hsdata_ff[4]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[50] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[50]),
        .Q(src_hsdata_ff[50]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[51] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[51]),
        .Q(src_hsdata_ff[51]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[52] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[52]),
        .Q(src_hsdata_ff[52]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[53] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[53]),
        .Q(src_hsdata_ff[53]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[54] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[54]),
        .Q(src_hsdata_ff[54]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[55] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[55]),
        .Q(src_hsdata_ff[55]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[56] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[56]),
        .Q(src_hsdata_ff[56]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[57] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[57]),
        .Q(src_hsdata_ff[57]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[58] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[58]),
        .Q(src_hsdata_ff[58]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[59] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[59]),
        .Q(src_hsdata_ff[59]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[5] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[5]),
        .Q(src_hsdata_ff[5]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[60] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[60]),
        .Q(src_hsdata_ff[60]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[61] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[61]),
        .Q(src_hsdata_ff[61]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[62] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[62]),
        .Q(src_hsdata_ff[62]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[63] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[63]),
        .Q(src_hsdata_ff[63]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[64] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[64]),
        .Q(src_hsdata_ff[64]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[6] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[6]),
        .Q(src_hsdata_ff[6]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[7] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[7]),
        .Q(src_hsdata_ff[7]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[8] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[8]),
        .Q(src_hsdata_ff[8]),
        .R(1'b0));
  FDRE \src_hsdata_ff_reg[9] 
       (.C(src_clk),
        .CE(p_0_in),
        .D(src_in[9]),
        .Q(src_hsdata_ff[9]),
        .R(1'b0));
  FDRE src_sendd_ff_reg
       (.C(src_clk),
        .CE(1'b1),
        .D(src_send),
        .Q(src_sendd_ff),
        .R(1'b0));
  (* DEST_SYNC_FF = "2" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_INPUT_REG = "0" *) 
  (* VERSION = "0" *) 
  (* XPM_CDC = "SINGLE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_single__9 xpm_cdc_single_dest2src_inst
       (.dest_clk(src_clk),
        .dest_out(src_rcv),
        .src_clk(dest_clk),
        .src_in(dest_ack));
  (* DEST_SYNC_FF = "2" *) 
  (* INIT_SYNC_FF = "0" *) 
  (* SIM_ASSERT_CHK = "0" *) 
  (* SRC_INPUT_REG = "0" *) 
  (* VERSION = "0" *) 
  (* XPM_CDC = "SINGLE" *) 
  (* XPM_MODULE = "TRUE" *) 
  xpm_cdc_single__8 xpm_cdc_single_src2dest_inst
       (.dest_clk(dest_clk),
        .dest_out(dest_req_nxt),
        .src_clk(src_clk),
        .src_in(src_sendd_ff));
endmodule

(* DEST_SYNC_FF = "2" *) (* INIT_SYNC_FF = "0" *) (* SIM_ASSERT_CHK = "0" *) 
(* SRC_INPUT_REG = "0" *) (* VERSION = "0" *) (* XPM_MODULE = "TRUE" *) 
(* xpm_cdc = "SINGLE" *) 
module xpm_cdc_single
   (src_clk,
    src_in,
    dest_clk,
    dest_out);
  input src_clk;
  input src_in;
  input dest_clk;
  output dest_out;

  wire dest_clk;
  wire src_in;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "SINGLE" *) wire [1:0]syncstages_ff;

  assign dest_out = syncstages_ff[1];
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SINGLE" *) 
  FDRE \syncstages_ff_reg[0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(src_in),
        .Q(syncstages_ff[0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SINGLE" *) 
  FDRE \syncstages_ff_reg[1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(syncstages_ff[0]),
        .Q(syncstages_ff[1]),
        .R(1'b0));
endmodule

(* DEST_SYNC_FF = "2" *) (* INIT_SYNC_FF = "0" *) (* ORIG_REF_NAME = "xpm_cdc_single" *) 
(* SIM_ASSERT_CHK = "0" *) (* SRC_INPUT_REG = "0" *) (* VERSION = "0" *) 
(* XPM_MODULE = "TRUE" *) (* xpm_cdc = "SINGLE" *) 
module xpm_cdc_single__10
   (src_clk,
    src_in,
    dest_clk,
    dest_out);
  input src_clk;
  input src_in;
  input dest_clk;
  output dest_out;

  wire dest_clk;
  wire src_in;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "SINGLE" *) wire [1:0]syncstages_ff;

  assign dest_out = syncstages_ff[1];
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SINGLE" *) 
  FDRE \syncstages_ff_reg[0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(src_in),
        .Q(syncstages_ff[0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SINGLE" *) 
  FDRE \syncstages_ff_reg[1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(syncstages_ff[0]),
        .Q(syncstages_ff[1]),
        .R(1'b0));
endmodule

(* DEST_SYNC_FF = "2" *) (* INIT_SYNC_FF = "0" *) (* ORIG_REF_NAME = "xpm_cdc_single" *) 
(* SIM_ASSERT_CHK = "0" *) (* SRC_INPUT_REG = "0" *) (* VERSION = "0" *) 
(* XPM_MODULE = "TRUE" *) (* xpm_cdc = "SINGLE" *) 
module xpm_cdc_single__6
   (src_clk,
    src_in,
    dest_clk,
    dest_out);
  input src_clk;
  input src_in;
  input dest_clk;
  output dest_out;

  wire dest_clk;
  wire src_in;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "SINGLE" *) wire [1:0]syncstages_ff;

  assign dest_out = syncstages_ff[1];
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SINGLE" *) 
  FDRE \syncstages_ff_reg[0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(src_in),
        .Q(syncstages_ff[0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SINGLE" *) 
  FDRE \syncstages_ff_reg[1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(syncstages_ff[0]),
        .Q(syncstages_ff[1]),
        .R(1'b0));
endmodule

(* DEST_SYNC_FF = "2" *) (* INIT_SYNC_FF = "0" *) (* ORIG_REF_NAME = "xpm_cdc_single" *) 
(* SIM_ASSERT_CHK = "0" *) (* SRC_INPUT_REG = "0" *) (* VERSION = "0" *) 
(* XPM_MODULE = "TRUE" *) (* xpm_cdc = "SINGLE" *) 
module xpm_cdc_single__7
   (src_clk,
    src_in,
    dest_clk,
    dest_out);
  input src_clk;
  input src_in;
  input dest_clk;
  output dest_out;

  wire dest_clk;
  wire src_in;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "SINGLE" *) wire [1:0]syncstages_ff;

  assign dest_out = syncstages_ff[1];
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SINGLE" *) 
  FDRE \syncstages_ff_reg[0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(src_in),
        .Q(syncstages_ff[0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SINGLE" *) 
  FDRE \syncstages_ff_reg[1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(syncstages_ff[0]),
        .Q(syncstages_ff[1]),
        .R(1'b0));
endmodule

(* DEST_SYNC_FF = "2" *) (* INIT_SYNC_FF = "0" *) (* ORIG_REF_NAME = "xpm_cdc_single" *) 
(* SIM_ASSERT_CHK = "0" *) (* SRC_INPUT_REG = "0" *) (* VERSION = "0" *) 
(* XPM_MODULE = "TRUE" *) (* xpm_cdc = "SINGLE" *) 
module xpm_cdc_single__8
   (src_clk,
    src_in,
    dest_clk,
    dest_out);
  input src_clk;
  input src_in;
  input dest_clk;
  output dest_out;

  wire dest_clk;
  wire src_in;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "SINGLE" *) wire [1:0]syncstages_ff;

  assign dest_out = syncstages_ff[1];
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SINGLE" *) 
  FDRE \syncstages_ff_reg[0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(src_in),
        .Q(syncstages_ff[0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SINGLE" *) 
  FDRE \syncstages_ff_reg[1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(syncstages_ff[0]),
        .Q(syncstages_ff[1]),
        .R(1'b0));
endmodule

(* DEST_SYNC_FF = "2" *) (* INIT_SYNC_FF = "0" *) (* ORIG_REF_NAME = "xpm_cdc_single" *) 
(* SIM_ASSERT_CHK = "0" *) (* SRC_INPUT_REG = "0" *) (* VERSION = "0" *) 
(* XPM_MODULE = "TRUE" *) (* xpm_cdc = "SINGLE" *) 
module xpm_cdc_single__9
   (src_clk,
    src_in,
    dest_clk,
    dest_out);
  input src_clk;
  input src_in;
  input dest_clk;
  output dest_out;

  wire dest_clk;
  wire src_in;
  (* RTL_KEEP = "true" *) (* async_reg = "true" *) (* xpm_cdc = "SINGLE" *) wire [1:0]syncstages_ff;

  assign dest_out = syncstages_ff[1];
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SINGLE" *) 
  FDRE \syncstages_ff_reg[0] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(src_in),
        .Q(syncstages_ff[0]),
        .R(1'b0));
  (* ASYNC_REG *) 
  (* KEEP = "true" *) 
  (* XPM_CDC = "SINGLE" *) 
  FDRE \syncstages_ff_reg[1] 
       (.C(dest_clk),
        .CE(1'b1),
        .D(syncstages_ff[0]),
        .Q(syncstages_ff[1]),
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
