// Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2018.3 (lin64) Build 2405991 Thu Dec  6 23:36:41 MST 2018
// Date        : Wed Nov 18 10:58:15 2020
// Host        : bender.ad.sklk.us running 64-bit Ubuntu 16.04.6 LTS
// Command     : write_verilog -force -mode funcsim twbw_packer_sim.v
// Design      : twbw_packer
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7z030sbg485-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* HDR_XFERS = "2" *) 
(* NotValidForBitStream *)
module twbw_packer
   (clk,
    rst,
    in_tctrl,
    in_tdata,
    in_tlast,
    in_tvalid,
    in_tready,
    out_tdata,
    out_tkeep,
    out_tlast,
    out_tvalid,
    out_tready);
  input clk;
  input rst;
  input [7:0]in_tctrl;
  input [63:0]in_tdata;
  input in_tlast;
  input in_tvalid;
  output in_tready;
  output [63:0]out_tdata;
  output [7:0]out_tkeep;
  output out_tlast;
  output out_tvalid;
  input out_tready;

  wire \<const1> ;
  wire \FSM_sequential_state[0]_i_1_n_0 ;
  wire \FSM_sequential_state[0]_i_2_n_0 ;
  wire \FSM_sequential_state[1]_i_1_n_0 ;
  wire \FSM_sequential_state[1]_i_2_n_0 ;
  wire \FSM_sequential_state[2]_i_1_n_0 ;
  wire \FSM_sequential_state[2]_i_2_n_0 ;
  wire \FSM_sequential_state[2]_i_3_n_0 ;
  wire \FSM_sequential_state[2]_i_4_n_0 ;
  wire \FSM_sequential_state[2]_i_5_n_0 ;
  wire \FSM_sequential_state[2]_i_6_n_0 ;
  wire [2:0]bytes_occ;
  wire clk;
  wire [7:0]data3;
  wire data_shift;
  wire \data_shift[0]_i_1_n_0 ;
  wire \data_shift[0]_i_2_n_0 ;
  wire \data_shift[0]_i_3_n_0 ;
  wire \data_shift[1]_i_1_n_0 ;
  wire \data_shift[1]_i_2_n_0 ;
  wire \data_shift[2]_i_1_n_0 ;
  wire \data_shift[2]_i_2_n_0 ;
  wire \data_shift[2]_i_4_n_0 ;
  wire \data_shift[3]_i_1_n_0 ;
  wire \data_shift[3]_i_2_n_0 ;
  wire \data_shift[4]_i_1_n_0 ;
  wire \data_shift[4]_i_2_n_0 ;
  wire \data_shift[5]_i_1_n_0 ;
  wire \data_shift[5]_i_2_n_0 ;
  wire \data_shift[6]_i_1_n_0 ;
  wire \data_shift[6]_i_2_n_0 ;
  wire \data_shift[7]_i_2_n_0 ;
  wire \data_shift[7]_i_3_n_0 ;
  wire [7:1]data_shift_next;
  wire \data_shift_reg_n_0_[7] ;
  wire extra_cycle_i_1_n_0;
  wire extra_cycle_i_2_n_0;
  wire extra_cycle_i_3_n_0;
  wire extra_cycle_reg_n_0;
  wire \hdr_shift[2]_i_1_n_0 ;
  wire \hdr_shift_reg_n_0_[2] ;
  wire [63:32]in14;
  wire [7:1]in8;
  wire [7:0]in_tctrl;
  wire [63:0]in_tdata;
  wire in_tlast;
  wire in_tready;
  wire in_tvalid;
  wire [63:0]out_tdata;
  wire \out_tdata[0]_INST_0_i_1_n_0 ;
  wire \out_tdata[0]_INST_0_i_2_n_0 ;
  wire \out_tdata[0]_INST_0_i_4_n_0 ;
  wire \out_tdata[0]_INST_0_i_5_n_0 ;
  wire \out_tdata[0]_INST_0_i_6_n_0 ;
  wire \out_tdata[0]_INST_0_i_7_n_0 ;
  wire \out_tdata[0]_INST_0_i_8_n_0 ;
  wire \out_tdata[10]_INST_0_i_1_n_0 ;
  wire \out_tdata[10]_INST_0_i_2_n_0 ;
  wire \out_tdata[10]_INST_0_i_4_n_0 ;
  wire \out_tdata[10]_INST_0_i_5_n_0 ;
  wire \out_tdata[10]_INST_0_i_6_n_0 ;
  wire \out_tdata[10]_INST_0_i_7_n_0 ;
  wire \out_tdata[10]_INST_0_i_8_n_0 ;
  wire \out_tdata[11]_INST_0_i_1_n_0 ;
  wire \out_tdata[11]_INST_0_i_2_n_0 ;
  wire \out_tdata[11]_INST_0_i_4_n_0 ;
  wire \out_tdata[11]_INST_0_i_5_n_0 ;
  wire \out_tdata[11]_INST_0_i_6_n_0 ;
  wire \out_tdata[11]_INST_0_i_7_n_0 ;
  wire \out_tdata[11]_INST_0_i_8_n_0 ;
  wire \out_tdata[12]_INST_0_i_1_n_0 ;
  wire \out_tdata[12]_INST_0_i_2_n_0 ;
  wire \out_tdata[12]_INST_0_i_4_n_0 ;
  wire \out_tdata[12]_INST_0_i_5_n_0 ;
  wire \out_tdata[12]_INST_0_i_6_n_0 ;
  wire \out_tdata[12]_INST_0_i_7_n_0 ;
  wire \out_tdata[12]_INST_0_i_8_n_0 ;
  wire \out_tdata[13]_INST_0_i_1_n_0 ;
  wire \out_tdata[13]_INST_0_i_2_n_0 ;
  wire \out_tdata[13]_INST_0_i_4_n_0 ;
  wire \out_tdata[13]_INST_0_i_5_n_0 ;
  wire \out_tdata[13]_INST_0_i_6_n_0 ;
  wire \out_tdata[13]_INST_0_i_7_n_0 ;
  wire \out_tdata[13]_INST_0_i_8_n_0 ;
  wire \out_tdata[14]_INST_0_i_1_n_0 ;
  wire \out_tdata[14]_INST_0_i_2_n_0 ;
  wire \out_tdata[14]_INST_0_i_4_n_0 ;
  wire \out_tdata[14]_INST_0_i_5_n_0 ;
  wire \out_tdata[14]_INST_0_i_6_n_0 ;
  wire \out_tdata[14]_INST_0_i_7_n_0 ;
  wire \out_tdata[14]_INST_0_i_8_n_0 ;
  wire \out_tdata[15]_INST_0_i_1_n_0 ;
  wire \out_tdata[15]_INST_0_i_2_n_0 ;
  wire \out_tdata[15]_INST_0_i_4_n_0 ;
  wire \out_tdata[15]_INST_0_i_5_n_0 ;
  wire \out_tdata[15]_INST_0_i_6_n_0 ;
  wire \out_tdata[15]_INST_0_i_7_n_0 ;
  wire \out_tdata[15]_INST_0_i_8_n_0 ;
  wire \out_tdata[16]_INST_0_i_1_n_0 ;
  wire \out_tdata[16]_INST_0_i_2_n_0 ;
  wire \out_tdata[16]_INST_0_i_4_n_0 ;
  wire \out_tdata[16]_INST_0_i_5_n_0 ;
  wire \out_tdata[16]_INST_0_i_6_n_0 ;
  wire \out_tdata[16]_INST_0_i_7_n_0 ;
  wire \out_tdata[16]_INST_0_i_8_n_0 ;
  wire \out_tdata[17]_INST_0_i_1_n_0 ;
  wire \out_tdata[17]_INST_0_i_2_n_0 ;
  wire \out_tdata[17]_INST_0_i_4_n_0 ;
  wire \out_tdata[17]_INST_0_i_5_n_0 ;
  wire \out_tdata[17]_INST_0_i_6_n_0 ;
  wire \out_tdata[17]_INST_0_i_7_n_0 ;
  wire \out_tdata[17]_INST_0_i_8_n_0 ;
  wire \out_tdata[18]_INST_0_i_1_n_0 ;
  wire \out_tdata[18]_INST_0_i_2_n_0 ;
  wire \out_tdata[18]_INST_0_i_4_n_0 ;
  wire \out_tdata[18]_INST_0_i_5_n_0 ;
  wire \out_tdata[18]_INST_0_i_6_n_0 ;
  wire \out_tdata[18]_INST_0_i_7_n_0 ;
  wire \out_tdata[18]_INST_0_i_8_n_0 ;
  wire \out_tdata[19]_INST_0_i_1_n_0 ;
  wire \out_tdata[19]_INST_0_i_2_n_0 ;
  wire \out_tdata[19]_INST_0_i_4_n_0 ;
  wire \out_tdata[19]_INST_0_i_5_n_0 ;
  wire \out_tdata[19]_INST_0_i_6_n_0 ;
  wire \out_tdata[19]_INST_0_i_7_n_0 ;
  wire \out_tdata[19]_INST_0_i_8_n_0 ;
  wire \out_tdata[1]_INST_0_i_1_n_0 ;
  wire \out_tdata[1]_INST_0_i_2_n_0 ;
  wire \out_tdata[1]_INST_0_i_4_n_0 ;
  wire \out_tdata[1]_INST_0_i_5_n_0 ;
  wire \out_tdata[1]_INST_0_i_6_n_0 ;
  wire \out_tdata[1]_INST_0_i_7_n_0 ;
  wire \out_tdata[1]_INST_0_i_8_n_0 ;
  wire \out_tdata[20]_INST_0_i_1_n_0 ;
  wire \out_tdata[20]_INST_0_i_2_n_0 ;
  wire \out_tdata[20]_INST_0_i_4_n_0 ;
  wire \out_tdata[20]_INST_0_i_5_n_0 ;
  wire \out_tdata[20]_INST_0_i_6_n_0 ;
  wire \out_tdata[20]_INST_0_i_7_n_0 ;
  wire \out_tdata[20]_INST_0_i_8_n_0 ;
  wire \out_tdata[21]_INST_0_i_1_n_0 ;
  wire \out_tdata[21]_INST_0_i_2_n_0 ;
  wire \out_tdata[21]_INST_0_i_4_n_0 ;
  wire \out_tdata[21]_INST_0_i_5_n_0 ;
  wire \out_tdata[21]_INST_0_i_6_n_0 ;
  wire \out_tdata[21]_INST_0_i_7_n_0 ;
  wire \out_tdata[21]_INST_0_i_8_n_0 ;
  wire \out_tdata[22]_INST_0_i_1_n_0 ;
  wire \out_tdata[22]_INST_0_i_2_n_0 ;
  wire \out_tdata[22]_INST_0_i_4_n_0 ;
  wire \out_tdata[22]_INST_0_i_5_n_0 ;
  wire \out_tdata[22]_INST_0_i_6_n_0 ;
  wire \out_tdata[22]_INST_0_i_7_n_0 ;
  wire \out_tdata[22]_INST_0_i_8_n_0 ;
  wire \out_tdata[23]_INST_0_i_1_n_0 ;
  wire \out_tdata[23]_INST_0_i_2_n_0 ;
  wire \out_tdata[23]_INST_0_i_4_n_0 ;
  wire \out_tdata[23]_INST_0_i_5_n_0 ;
  wire \out_tdata[23]_INST_0_i_6_n_0 ;
  wire \out_tdata[23]_INST_0_i_7_n_0 ;
  wire \out_tdata[23]_INST_0_i_8_n_0 ;
  wire \out_tdata[24]_INST_0_i_1_n_0 ;
  wire \out_tdata[24]_INST_0_i_2_n_0 ;
  wire \out_tdata[24]_INST_0_i_4_n_0 ;
  wire \out_tdata[24]_INST_0_i_5_n_0 ;
  wire \out_tdata[24]_INST_0_i_6_n_0 ;
  wire \out_tdata[24]_INST_0_i_7_n_0 ;
  wire \out_tdata[24]_INST_0_i_8_n_0 ;
  wire \out_tdata[25]_INST_0_i_1_n_0 ;
  wire \out_tdata[25]_INST_0_i_2_n_0 ;
  wire \out_tdata[25]_INST_0_i_4_n_0 ;
  wire \out_tdata[25]_INST_0_i_5_n_0 ;
  wire \out_tdata[25]_INST_0_i_6_n_0 ;
  wire \out_tdata[25]_INST_0_i_7_n_0 ;
  wire \out_tdata[25]_INST_0_i_8_n_0 ;
  wire \out_tdata[26]_INST_0_i_1_n_0 ;
  wire \out_tdata[26]_INST_0_i_2_n_0 ;
  wire \out_tdata[26]_INST_0_i_4_n_0 ;
  wire \out_tdata[26]_INST_0_i_5_n_0 ;
  wire \out_tdata[26]_INST_0_i_6_n_0 ;
  wire \out_tdata[26]_INST_0_i_7_n_0 ;
  wire \out_tdata[26]_INST_0_i_8_n_0 ;
  wire \out_tdata[27]_INST_0_i_1_n_0 ;
  wire \out_tdata[27]_INST_0_i_2_n_0 ;
  wire \out_tdata[27]_INST_0_i_4_n_0 ;
  wire \out_tdata[27]_INST_0_i_5_n_0 ;
  wire \out_tdata[27]_INST_0_i_6_n_0 ;
  wire \out_tdata[27]_INST_0_i_7_n_0 ;
  wire \out_tdata[27]_INST_0_i_8_n_0 ;
  wire \out_tdata[28]_INST_0_i_1_n_0 ;
  wire \out_tdata[28]_INST_0_i_2_n_0 ;
  wire \out_tdata[28]_INST_0_i_4_n_0 ;
  wire \out_tdata[28]_INST_0_i_5_n_0 ;
  wire \out_tdata[28]_INST_0_i_6_n_0 ;
  wire \out_tdata[28]_INST_0_i_7_n_0 ;
  wire \out_tdata[28]_INST_0_i_8_n_0 ;
  wire \out_tdata[29]_INST_0_i_1_n_0 ;
  wire \out_tdata[29]_INST_0_i_2_n_0 ;
  wire \out_tdata[29]_INST_0_i_4_n_0 ;
  wire \out_tdata[29]_INST_0_i_5_n_0 ;
  wire \out_tdata[29]_INST_0_i_6_n_0 ;
  wire \out_tdata[29]_INST_0_i_7_n_0 ;
  wire \out_tdata[29]_INST_0_i_8_n_0 ;
  wire \out_tdata[2]_INST_0_i_1_n_0 ;
  wire \out_tdata[2]_INST_0_i_2_n_0 ;
  wire \out_tdata[2]_INST_0_i_4_n_0 ;
  wire \out_tdata[2]_INST_0_i_5_n_0 ;
  wire \out_tdata[2]_INST_0_i_6_n_0 ;
  wire \out_tdata[2]_INST_0_i_7_n_0 ;
  wire \out_tdata[2]_INST_0_i_8_n_0 ;
  wire \out_tdata[30]_INST_0_i_1_n_0 ;
  wire \out_tdata[30]_INST_0_i_2_n_0 ;
  wire \out_tdata[30]_INST_0_i_4_n_0 ;
  wire \out_tdata[30]_INST_0_i_5_n_0 ;
  wire \out_tdata[30]_INST_0_i_6_n_0 ;
  wire \out_tdata[30]_INST_0_i_7_n_0 ;
  wire \out_tdata[30]_INST_0_i_8_n_0 ;
  wire \out_tdata[31]_INST_0_i_1_n_0 ;
  wire \out_tdata[31]_INST_0_i_2_n_0 ;
  wire \out_tdata[31]_INST_0_i_4_n_0 ;
  wire \out_tdata[31]_INST_0_i_5_n_0 ;
  wire \out_tdata[31]_INST_0_i_6_n_0 ;
  wire \out_tdata[31]_INST_0_i_7_n_0 ;
  wire \out_tdata[31]_INST_0_i_8_n_0 ;
  wire \out_tdata[32]_INST_0_i_2_n_0 ;
  wire \out_tdata[32]_INST_0_i_3_n_0 ;
  wire \out_tdata[32]_INST_0_i_4_n_0 ;
  wire \out_tdata[32]_INST_0_i_5_n_0 ;
  wire \out_tdata[32]_INST_0_i_7_n_0 ;
  wire \out_tdata[32]_INST_0_i_8_n_0 ;
  wire \out_tdata[32]_INST_0_i_9_n_0 ;
  wire \out_tdata[33]_INST_0_i_2_n_0 ;
  wire \out_tdata[33]_INST_0_i_3_n_0 ;
  wire \out_tdata[33]_INST_0_i_4_n_0 ;
  wire \out_tdata[33]_INST_0_i_5_n_0 ;
  wire \out_tdata[33]_INST_0_i_7_n_0 ;
  wire \out_tdata[33]_INST_0_i_8_n_0 ;
  wire \out_tdata[33]_INST_0_i_9_n_0 ;
  wire \out_tdata[34]_INST_0_i_2_n_0 ;
  wire \out_tdata[34]_INST_0_i_3_n_0 ;
  wire \out_tdata[34]_INST_0_i_4_n_0 ;
  wire \out_tdata[34]_INST_0_i_5_n_0 ;
  wire \out_tdata[34]_INST_0_i_7_n_0 ;
  wire \out_tdata[34]_INST_0_i_8_n_0 ;
  wire \out_tdata[34]_INST_0_i_9_n_0 ;
  wire \out_tdata[35]_INST_0_i_2_n_0 ;
  wire \out_tdata[35]_INST_0_i_3_n_0 ;
  wire \out_tdata[35]_INST_0_i_4_n_0 ;
  wire \out_tdata[35]_INST_0_i_5_n_0 ;
  wire \out_tdata[35]_INST_0_i_7_n_0 ;
  wire \out_tdata[35]_INST_0_i_8_n_0 ;
  wire \out_tdata[35]_INST_0_i_9_n_0 ;
  wire \out_tdata[36]_INST_0_i_2_n_0 ;
  wire \out_tdata[36]_INST_0_i_3_n_0 ;
  wire \out_tdata[36]_INST_0_i_4_n_0 ;
  wire \out_tdata[36]_INST_0_i_5_n_0 ;
  wire \out_tdata[36]_INST_0_i_7_n_0 ;
  wire \out_tdata[36]_INST_0_i_8_n_0 ;
  wire \out_tdata[36]_INST_0_i_9_n_0 ;
  wire \out_tdata[37]_INST_0_i_2_n_0 ;
  wire \out_tdata[37]_INST_0_i_3_n_0 ;
  wire \out_tdata[37]_INST_0_i_4_n_0 ;
  wire \out_tdata[37]_INST_0_i_5_n_0 ;
  wire \out_tdata[37]_INST_0_i_7_n_0 ;
  wire \out_tdata[37]_INST_0_i_8_n_0 ;
  wire \out_tdata[37]_INST_0_i_9_n_0 ;
  wire \out_tdata[38]_INST_0_i_2_n_0 ;
  wire \out_tdata[38]_INST_0_i_3_n_0 ;
  wire \out_tdata[38]_INST_0_i_4_n_0 ;
  wire \out_tdata[38]_INST_0_i_5_n_0 ;
  wire \out_tdata[38]_INST_0_i_7_n_0 ;
  wire \out_tdata[38]_INST_0_i_8_n_0 ;
  wire \out_tdata[38]_INST_0_i_9_n_0 ;
  wire \out_tdata[39]_INST_0_i_2_n_0 ;
  wire \out_tdata[39]_INST_0_i_3_n_0 ;
  wire \out_tdata[39]_INST_0_i_4_n_0 ;
  wire \out_tdata[39]_INST_0_i_5_n_0 ;
  wire \out_tdata[39]_INST_0_i_7_n_0 ;
  wire \out_tdata[39]_INST_0_i_8_n_0 ;
  wire \out_tdata[39]_INST_0_i_9_n_0 ;
  wire \out_tdata[3]_INST_0_i_1_n_0 ;
  wire \out_tdata[3]_INST_0_i_2_n_0 ;
  wire \out_tdata[3]_INST_0_i_4_n_0 ;
  wire \out_tdata[3]_INST_0_i_5_n_0 ;
  wire \out_tdata[3]_INST_0_i_6_n_0 ;
  wire \out_tdata[3]_INST_0_i_7_n_0 ;
  wire \out_tdata[3]_INST_0_i_8_n_0 ;
  wire \out_tdata[40]_INST_0_i_1_n_0 ;
  wire \out_tdata[40]_INST_0_i_2_n_0 ;
  wire \out_tdata[40]_INST_0_i_4_n_0 ;
  wire \out_tdata[40]_INST_0_i_5_n_0 ;
  wire \out_tdata[40]_INST_0_i_7_n_0 ;
  wire \out_tdata[40]_INST_0_i_8_n_0 ;
  wire \out_tdata[40]_INST_0_i_9_n_0 ;
  wire \out_tdata[41]_INST_0_i_1_n_0 ;
  wire \out_tdata[41]_INST_0_i_2_n_0 ;
  wire \out_tdata[41]_INST_0_i_4_n_0 ;
  wire \out_tdata[41]_INST_0_i_5_n_0 ;
  wire \out_tdata[41]_INST_0_i_7_n_0 ;
  wire \out_tdata[41]_INST_0_i_8_n_0 ;
  wire \out_tdata[41]_INST_0_i_9_n_0 ;
  wire \out_tdata[42]_INST_0_i_1_n_0 ;
  wire \out_tdata[42]_INST_0_i_2_n_0 ;
  wire \out_tdata[42]_INST_0_i_4_n_0 ;
  wire \out_tdata[42]_INST_0_i_5_n_0 ;
  wire \out_tdata[42]_INST_0_i_7_n_0 ;
  wire \out_tdata[42]_INST_0_i_8_n_0 ;
  wire \out_tdata[42]_INST_0_i_9_n_0 ;
  wire \out_tdata[43]_INST_0_i_1_n_0 ;
  wire \out_tdata[43]_INST_0_i_2_n_0 ;
  wire \out_tdata[43]_INST_0_i_4_n_0 ;
  wire \out_tdata[43]_INST_0_i_5_n_0 ;
  wire \out_tdata[43]_INST_0_i_7_n_0 ;
  wire \out_tdata[43]_INST_0_i_8_n_0 ;
  wire \out_tdata[43]_INST_0_i_9_n_0 ;
  wire \out_tdata[44]_INST_0_i_1_n_0 ;
  wire \out_tdata[44]_INST_0_i_2_n_0 ;
  wire \out_tdata[44]_INST_0_i_4_n_0 ;
  wire \out_tdata[44]_INST_0_i_5_n_0 ;
  wire \out_tdata[44]_INST_0_i_7_n_0 ;
  wire \out_tdata[44]_INST_0_i_8_n_0 ;
  wire \out_tdata[44]_INST_0_i_9_n_0 ;
  wire \out_tdata[45]_INST_0_i_1_n_0 ;
  wire \out_tdata[45]_INST_0_i_2_n_0 ;
  wire \out_tdata[45]_INST_0_i_4_n_0 ;
  wire \out_tdata[45]_INST_0_i_5_n_0 ;
  wire \out_tdata[45]_INST_0_i_7_n_0 ;
  wire \out_tdata[45]_INST_0_i_8_n_0 ;
  wire \out_tdata[45]_INST_0_i_9_n_0 ;
  wire \out_tdata[46]_INST_0_i_1_n_0 ;
  wire \out_tdata[46]_INST_0_i_2_n_0 ;
  wire \out_tdata[46]_INST_0_i_4_n_0 ;
  wire \out_tdata[46]_INST_0_i_5_n_0 ;
  wire \out_tdata[46]_INST_0_i_7_n_0 ;
  wire \out_tdata[46]_INST_0_i_8_n_0 ;
  wire \out_tdata[46]_INST_0_i_9_n_0 ;
  wire \out_tdata[47]_INST_0_i_1_n_0 ;
  wire \out_tdata[47]_INST_0_i_2_n_0 ;
  wire \out_tdata[47]_INST_0_i_4_n_0 ;
  wire \out_tdata[47]_INST_0_i_5_n_0 ;
  wire \out_tdata[47]_INST_0_i_7_n_0 ;
  wire \out_tdata[47]_INST_0_i_8_n_0 ;
  wire \out_tdata[47]_INST_0_i_9_n_0 ;
  wire \out_tdata[48]_INST_0_i_2_n_0 ;
  wire \out_tdata[48]_INST_0_i_3_n_0 ;
  wire \out_tdata[48]_INST_0_i_4_n_0 ;
  wire \out_tdata[48]_INST_0_i_6_n_0 ;
  wire \out_tdata[48]_INST_0_i_7_n_0 ;
  wire \out_tdata[48]_INST_0_i_8_n_0 ;
  wire \out_tdata[49]_INST_0_i_2_n_0 ;
  wire \out_tdata[49]_INST_0_i_3_n_0 ;
  wire \out_tdata[49]_INST_0_i_4_n_0 ;
  wire \out_tdata[49]_INST_0_i_6_n_0 ;
  wire \out_tdata[49]_INST_0_i_7_n_0 ;
  wire \out_tdata[49]_INST_0_i_8_n_0 ;
  wire \out_tdata[4]_INST_0_i_1_n_0 ;
  wire \out_tdata[4]_INST_0_i_2_n_0 ;
  wire \out_tdata[4]_INST_0_i_4_n_0 ;
  wire \out_tdata[4]_INST_0_i_5_n_0 ;
  wire \out_tdata[4]_INST_0_i_6_n_0 ;
  wire \out_tdata[4]_INST_0_i_7_n_0 ;
  wire \out_tdata[4]_INST_0_i_8_n_0 ;
  wire \out_tdata[50]_INST_0_i_2_n_0 ;
  wire \out_tdata[50]_INST_0_i_3_n_0 ;
  wire \out_tdata[50]_INST_0_i_4_n_0 ;
  wire \out_tdata[50]_INST_0_i_6_n_0 ;
  wire \out_tdata[50]_INST_0_i_7_n_0 ;
  wire \out_tdata[50]_INST_0_i_8_n_0 ;
  wire \out_tdata[51]_INST_0_i_2_n_0 ;
  wire \out_tdata[51]_INST_0_i_3_n_0 ;
  wire \out_tdata[51]_INST_0_i_4_n_0 ;
  wire \out_tdata[51]_INST_0_i_6_n_0 ;
  wire \out_tdata[51]_INST_0_i_7_n_0 ;
  wire \out_tdata[51]_INST_0_i_8_n_0 ;
  wire \out_tdata[52]_INST_0_i_2_n_0 ;
  wire \out_tdata[52]_INST_0_i_3_n_0 ;
  wire \out_tdata[52]_INST_0_i_4_n_0 ;
  wire \out_tdata[52]_INST_0_i_6_n_0 ;
  wire \out_tdata[52]_INST_0_i_7_n_0 ;
  wire \out_tdata[52]_INST_0_i_8_n_0 ;
  wire \out_tdata[53]_INST_0_i_2_n_0 ;
  wire \out_tdata[53]_INST_0_i_3_n_0 ;
  wire \out_tdata[53]_INST_0_i_4_n_0 ;
  wire \out_tdata[53]_INST_0_i_6_n_0 ;
  wire \out_tdata[53]_INST_0_i_7_n_0 ;
  wire \out_tdata[53]_INST_0_i_8_n_0 ;
  wire \out_tdata[54]_INST_0_i_2_n_0 ;
  wire \out_tdata[54]_INST_0_i_3_n_0 ;
  wire \out_tdata[54]_INST_0_i_4_n_0 ;
  wire \out_tdata[54]_INST_0_i_6_n_0 ;
  wire \out_tdata[54]_INST_0_i_7_n_0 ;
  wire \out_tdata[54]_INST_0_i_8_n_0 ;
  wire \out_tdata[55]_INST_0_i_2_n_0 ;
  wire \out_tdata[55]_INST_0_i_3_n_0 ;
  wire \out_tdata[55]_INST_0_i_4_n_0 ;
  wire \out_tdata[55]_INST_0_i_6_n_0 ;
  wire \out_tdata[55]_INST_0_i_7_n_0 ;
  wire \out_tdata[55]_INST_0_i_8_n_0 ;
  wire \out_tdata[56]_INST_0_i_2_n_0 ;
  wire \out_tdata[56]_INST_0_i_3_n_0 ;
  wire \out_tdata[56]_INST_0_i_4_n_0 ;
  wire \out_tdata[56]_INST_0_i_6_n_0 ;
  wire \out_tdata[56]_INST_0_i_7_n_0 ;
  wire \out_tdata[56]_INST_0_i_8_n_0 ;
  wire \out_tdata[57]_INST_0_i_2_n_0 ;
  wire \out_tdata[57]_INST_0_i_3_n_0 ;
  wire \out_tdata[57]_INST_0_i_4_n_0 ;
  wire \out_tdata[57]_INST_0_i_6_n_0 ;
  wire \out_tdata[57]_INST_0_i_7_n_0 ;
  wire \out_tdata[57]_INST_0_i_8_n_0 ;
  wire \out_tdata[58]_INST_0_i_2_n_0 ;
  wire \out_tdata[58]_INST_0_i_3_n_0 ;
  wire \out_tdata[58]_INST_0_i_4_n_0 ;
  wire \out_tdata[58]_INST_0_i_6_n_0 ;
  wire \out_tdata[58]_INST_0_i_7_n_0 ;
  wire \out_tdata[58]_INST_0_i_8_n_0 ;
  wire \out_tdata[59]_INST_0_i_2_n_0 ;
  wire \out_tdata[59]_INST_0_i_3_n_0 ;
  wire \out_tdata[59]_INST_0_i_4_n_0 ;
  wire \out_tdata[59]_INST_0_i_6_n_0 ;
  wire \out_tdata[59]_INST_0_i_7_n_0 ;
  wire \out_tdata[59]_INST_0_i_8_n_0 ;
  wire \out_tdata[5]_INST_0_i_1_n_0 ;
  wire \out_tdata[5]_INST_0_i_2_n_0 ;
  wire \out_tdata[5]_INST_0_i_4_n_0 ;
  wire \out_tdata[5]_INST_0_i_5_n_0 ;
  wire \out_tdata[5]_INST_0_i_6_n_0 ;
  wire \out_tdata[5]_INST_0_i_7_n_0 ;
  wire \out_tdata[5]_INST_0_i_8_n_0 ;
  wire \out_tdata[60]_INST_0_i_2_n_0 ;
  wire \out_tdata[60]_INST_0_i_3_n_0 ;
  wire \out_tdata[60]_INST_0_i_4_n_0 ;
  wire \out_tdata[60]_INST_0_i_6_n_0 ;
  wire \out_tdata[60]_INST_0_i_7_n_0 ;
  wire \out_tdata[60]_INST_0_i_8_n_0 ;
  wire \out_tdata[61]_INST_0_i_2_n_0 ;
  wire \out_tdata[61]_INST_0_i_3_n_0 ;
  wire \out_tdata[61]_INST_0_i_4_n_0 ;
  wire \out_tdata[61]_INST_0_i_6_n_0 ;
  wire \out_tdata[61]_INST_0_i_7_n_0 ;
  wire \out_tdata[61]_INST_0_i_8_n_0 ;
  wire \out_tdata[62]_INST_0_i_2_n_0 ;
  wire \out_tdata[62]_INST_0_i_3_n_0 ;
  wire \out_tdata[62]_INST_0_i_4_n_0 ;
  wire \out_tdata[62]_INST_0_i_6_n_0 ;
  wire \out_tdata[62]_INST_0_i_7_n_0 ;
  wire \out_tdata[62]_INST_0_i_8_n_0 ;
  wire \out_tdata[63]_INST_0_i_10_n_0 ;
  wire \out_tdata[63]_INST_0_i_11_n_0 ;
  wire \out_tdata[63]_INST_0_i_12_n_0 ;
  wire \out_tdata[63]_INST_0_i_13_n_0 ;
  wire \out_tdata[63]_INST_0_i_14_n_0 ;
  wire \out_tdata[63]_INST_0_i_15_n_0 ;
  wire \out_tdata[63]_INST_0_i_16_n_0 ;
  wire \out_tdata[63]_INST_0_i_1_n_0 ;
  wire \out_tdata[63]_INST_0_i_3_n_0 ;
  wire \out_tdata[63]_INST_0_i_4_n_0 ;
  wire \out_tdata[63]_INST_0_i_5_n_0 ;
  wire \out_tdata[63]_INST_0_i_6_n_0 ;
  wire \out_tdata[63]_INST_0_i_8_n_0 ;
  wire \out_tdata[63]_INST_0_i_9_n_0 ;
  wire \out_tdata[6]_INST_0_i_1_n_0 ;
  wire \out_tdata[6]_INST_0_i_2_n_0 ;
  wire \out_tdata[6]_INST_0_i_4_n_0 ;
  wire \out_tdata[6]_INST_0_i_5_n_0 ;
  wire \out_tdata[6]_INST_0_i_6_n_0 ;
  wire \out_tdata[6]_INST_0_i_7_n_0 ;
  wire \out_tdata[6]_INST_0_i_8_n_0 ;
  wire \out_tdata[7]_INST_0_i_1_n_0 ;
  wire \out_tdata[7]_INST_0_i_2_n_0 ;
  wire \out_tdata[7]_INST_0_i_4_n_0 ;
  wire \out_tdata[7]_INST_0_i_5_n_0 ;
  wire \out_tdata[7]_INST_0_i_6_n_0 ;
  wire \out_tdata[7]_INST_0_i_7_n_0 ;
  wire \out_tdata[7]_INST_0_i_8_n_0 ;
  wire \out_tdata[8]_INST_0_i_1_n_0 ;
  wire \out_tdata[8]_INST_0_i_2_n_0 ;
  wire \out_tdata[8]_INST_0_i_4_n_0 ;
  wire \out_tdata[8]_INST_0_i_5_n_0 ;
  wire \out_tdata[8]_INST_0_i_6_n_0 ;
  wire \out_tdata[8]_INST_0_i_7_n_0 ;
  wire \out_tdata[8]_INST_0_i_8_n_0 ;
  wire \out_tdata[9]_INST_0_i_1_n_0 ;
  wire \out_tdata[9]_INST_0_i_2_n_0 ;
  wire \out_tdata[9]_INST_0_i_4_n_0 ;
  wire \out_tdata[9]_INST_0_i_5_n_0 ;
  wire \out_tdata[9]_INST_0_i_6_n_0 ;
  wire \out_tdata[9]_INST_0_i_7_n_0 ;
  wire \out_tdata[9]_INST_0_i_8_n_0 ;
  wire [7:1]\^out_tkeep ;
  wire \out_tkeep[7]_INST_0_i_1_n_0 ;
  wire \out_tkeep[7]_INST_0_i_2_n_0 ;
  wire \out_tkeep[7]_INST_0_i_3_n_0 ;
  wire \out_tkeep[7]_INST_0_i_4_n_0 ;
  wire \out_tkeep[7]_INST_0_i_5_n_0 ;
  wire \out_tkeep[7]_INST_0_i_6_n_0 ;
  wire out_tlast;
  wire out_tlast_INST_0_i_1_n_0;
  wire out_tlast_INST_0_i_2_n_0;
  wire out_tlast_INST_0_i_3_n_0;
  wire out_tlast_INST_0_i_4_n_0;
  wire out_tready;
  wire out_tvalid;
  wire out_tvalid_INST_0_i_2_n_0;
  wire out_tvalid_INST_0_i_4_n_0;
  wire out_tvalid_INST_0_i_5_n_0;
  wire out_tvalid_INST_0_i_6_n_0;
  wire out_tvalid_i1;
  wire [15:0]p_1_in;
  wire [39:0]p_1_in__0;
  wire p_2_in15_out;
  wire [47:0]p_2_in__0;
  wire \packed_data[0]_i_1_n_0 ;
  wire \packed_data[10]_i_1_n_0 ;
  wire \packed_data[11]_i_1_n_0 ;
  wire \packed_data[12]_i_1_n_0 ;
  wire \packed_data[13]_i_1_n_0 ;
  wire \packed_data[14]_i_1_n_0 ;
  wire \packed_data[15]_i_1_n_0 ;
  wire \packed_data[16]_i_1_n_0 ;
  wire \packed_data[17]_i_1_n_0 ;
  wire \packed_data[18]_i_1_n_0 ;
  wire \packed_data[19]_i_1_n_0 ;
  wire \packed_data[1]_i_1_n_0 ;
  wire \packed_data[20]_i_1_n_0 ;
  wire \packed_data[21]_i_1_n_0 ;
  wire \packed_data[22]_i_1_n_0 ;
  wire \packed_data[23]_i_1_n_0 ;
  wire \packed_data[24]_i_1_n_0 ;
  wire \packed_data[25]_i_1_n_0 ;
  wire \packed_data[26]_i_1_n_0 ;
  wire \packed_data[27]_i_1_n_0 ;
  wire \packed_data[28]_i_1_n_0 ;
  wire \packed_data[29]_i_1_n_0 ;
  wire \packed_data[2]_i_1_n_0 ;
  wire \packed_data[30]_i_1_n_0 ;
  wire \packed_data[31]_i_1_n_0 ;
  wire \packed_data[32]_i_1_n_0 ;
  wire \packed_data[33]_i_1_n_0 ;
  wire \packed_data[34]_i_1_n_0 ;
  wire \packed_data[35]_i_1_n_0 ;
  wire \packed_data[36]_i_1_n_0 ;
  wire \packed_data[37]_i_1_n_0 ;
  wire \packed_data[38]_i_1_n_0 ;
  wire \packed_data[39]_i_1_n_0 ;
  wire \packed_data[3]_i_1_n_0 ;
  wire \packed_data[40]_i_1_n_0 ;
  wire \packed_data[41]_i_1_n_0 ;
  wire \packed_data[42]_i_1_n_0 ;
  wire \packed_data[43]_i_1_n_0 ;
  wire \packed_data[44]_i_1_n_0 ;
  wire \packed_data[45]_i_1_n_0 ;
  wire \packed_data[46]_i_1_n_0 ;
  wire \packed_data[47]_i_1_n_0 ;
  wire \packed_data[48]_i_1_n_0 ;
  wire \packed_data[49]_i_1_n_0 ;
  wire \packed_data[4]_i_1_n_0 ;
  wire \packed_data[50]_i_1_n_0 ;
  wire \packed_data[51]_i_1_n_0 ;
  wire \packed_data[52]_i_1_n_0 ;
  wire \packed_data[53]_i_1_n_0 ;
  wire \packed_data[54]_i_1_n_0 ;
  wire \packed_data[55]_i_1_n_0 ;
  wire \packed_data[56]_i_1_n_0 ;
  wire \packed_data[57]_i_1_n_0 ;
  wire \packed_data[58]_i_1_n_0 ;
  wire \packed_data[59]_i_1_n_0 ;
  wire \packed_data[5]_i_1_n_0 ;
  wire \packed_data[60]_i_1_n_0 ;
  wire \packed_data[61]_i_1_n_0 ;
  wire \packed_data[62]_i_1_n_0 ;
  wire \packed_data[63]_i_1_n_0 ;
  wire \packed_data[6]_i_1_n_0 ;
  wire \packed_data[7]_i_1_n_0 ;
  wire \packed_data[8]_i_1_n_0 ;
  wire \packed_data[9]_i_1_n_0 ;
  wire \packed_data_reg_n_0_[40] ;
  wire \packed_data_reg_n_0_[41] ;
  wire \packed_data_reg_n_0_[42] ;
  wire \packed_data_reg_n_0_[43] ;
  wire \packed_data_reg_n_0_[44] ;
  wire \packed_data_reg_n_0_[45] ;
  wire \packed_data_reg_n_0_[46] ;
  wire \packed_data_reg_n_0_[47] ;
  wire \packed_data_reg_n_0_[48] ;
  wire \packed_data_reg_n_0_[49] ;
  wire \packed_data_reg_n_0_[50] ;
  wire \packed_data_reg_n_0_[51] ;
  wire \packed_data_reg_n_0_[52] ;
  wire \packed_data_reg_n_0_[53] ;
  wire \packed_data_reg_n_0_[54] ;
  wire \packed_data_reg_n_0_[55] ;
  wire \packed_data_reg_n_0_[56] ;
  wire \packed_data_reg_n_0_[57] ;
  wire \packed_data_reg_n_0_[58] ;
  wire \packed_data_reg_n_0_[59] ;
  wire \packed_data_reg_n_0_[60] ;
  wire \packed_data_reg_n_0_[61] ;
  wire \packed_data_reg_n_0_[62] ;
  wire \packed_data_reg_n_0_[63] ;
  wire rst;
  wire \samp24_last_reg_n_0_[10] ;
  wire \samp24_last_reg_n_0_[11] ;
  wire \samp24_last_reg_n_0_[12] ;
  wire \samp24_last_reg_n_0_[13] ;
  wire \samp24_last_reg_n_0_[14] ;
  wire \samp24_last_reg_n_0_[15] ;
  wire \samp24_last_reg_n_0_[8] ;
  wire \samp24_last_reg_n_0_[9] ;
  wire [31:0]samp32;
  wire \samp48_last_reg_n_0_[24] ;
  wire \samp48_last_reg_n_0_[25] ;
  wire \samp48_last_reg_n_0_[26] ;
  wire \samp48_last_reg_n_0_[27] ;
  wire \samp48_last_reg_n_0_[28] ;
  wire \samp48_last_reg_n_0_[29] ;
  wire \samp48_last_reg_n_0_[30] ;
  wire \samp48_last_reg_n_0_[31] ;
  wire [2:0]state__0;
  wire xfer_cycle4;
  wire xfer_cycle6;

  assign out_tkeep[7:1] = \^out_tkeep [7:1];
  assign out_tkeep[0] = \<const1> ;
  LUT4 #(
    .INIT(16'hFFE2)) 
    \FSM_sequential_state[0]_i_1 
       (.I0(state__0[0]),
        .I1(\FSM_sequential_state[2]_i_2_n_0 ),
        .I2(\FSM_sequential_state[0]_i_2_n_0 ),
        .I3(rst),
        .O(\FSM_sequential_state[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h5555FFAB5555FFFF)) 
    \FSM_sequential_state[0]_i_2 
       (.I0(state__0[1]),
        .I1(in_tctrl[0]),
        .I2(in_tctrl[2]),
        .I3(in_tlast),
        .I4(state__0[2]),
        .I5(state__0[0]),
        .O(\FSM_sequential_state[0]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h00E2)) 
    \FSM_sequential_state[1]_i_1 
       (.I0(state__0[1]),
        .I1(\FSM_sequential_state[2]_i_2_n_0 ),
        .I2(\FSM_sequential_state[1]_i_2_n_0 ),
        .I3(rst),
        .O(\FSM_sequential_state[1]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000100000)) 
    \FSM_sequential_state[1]_i_2 
       (.I0(state__0[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(in_tctrl[2]),
        .I4(in_tctrl[1]),
        .I5(in_tlast),
        .O(\FSM_sequential_state[1]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h00E2)) 
    \FSM_sequential_state[2]_i_1 
       (.I0(state__0[2]),
        .I1(\FSM_sequential_state[2]_i_2_n_0 ),
        .I2(\FSM_sequential_state[2]_i_3_n_0 ),
        .I3(rst),
        .O(\FSM_sequential_state[2]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAFFAAEEFAAAAAEEA)) 
    \FSM_sequential_state[2]_i_2 
       (.I0(\FSM_sequential_state[2]_i_4_n_0 ),
        .I1(\FSM_sequential_state[2]_i_5_n_0 ),
        .I2(state__0[2]),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .I5(\FSM_sequential_state[2]_i_6_n_0 ),
        .O(\FSM_sequential_state[2]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000010)) 
    \FSM_sequential_state[2]_i_3 
       (.I0(state__0[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(in_tctrl[2]),
        .I4(in_tctrl[1]),
        .I5(in_tlast),
        .O(\FSM_sequential_state[2]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'h04000000)) 
    \FSM_sequential_state[2]_i_4 
       (.I0(\hdr_shift_reg_n_0_[2] ),
        .I1(in_tvalid),
        .I2(extra_cycle_reg_n_0),
        .I3(out_tready),
        .I4(\out_tdata[63]_INST_0_i_1_n_0 ),
        .O(\FSM_sequential_state[2]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT5 #(
    .INIT(32'hE400C000)) 
    \FSM_sequential_state[2]_i_5 
       (.I0(out_tvalid_i1),
        .I1(out_tlast),
        .I2(in_tvalid),
        .I3(out_tready),
        .I4(extra_cycle_reg_n_0),
        .O(\FSM_sequential_state[2]_i_5_n_0 ));
  LUT4 #(
    .INIT(16'h2000)) 
    \FSM_sequential_state[2]_i_6 
       (.I0(out_tready),
        .I1(extra_cycle_reg_n_0),
        .I2(in_tvalid),
        .I3(in_tlast),
        .O(\FSM_sequential_state[2]_i_6_n_0 ));
  (* FSM_ENCODED_STATES = "state_hdr:001,state_data_16:000,state_data_24:010,state_data_32:011,state_data_48:100,state_data_64:101" *) 
  FDRE \FSM_sequential_state_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\FSM_sequential_state[0]_i_1_n_0 ),
        .Q(state__0[0]),
        .R(1'b0));
  (* FSM_ENCODED_STATES = "state_hdr:001,state_data_16:000,state_data_24:010,state_data_32:011,state_data_48:100,state_data_64:101" *) 
  FDRE \FSM_sequential_state_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\FSM_sequential_state[1]_i_1_n_0 ),
        .Q(state__0[1]),
        .R(1'b0));
  (* FSM_ENCODED_STATES = "state_hdr:001,state_data_16:000,state_data_24:010,state_data_32:011,state_data_48:100,state_data_64:101" *) 
  FDRE \FSM_sequential_state_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\FSM_sequential_state[2]_i_1_n_0 ),
        .Q(state__0[2]),
        .R(1'b0));
  VCC VCC
       (.P(\<const1> ));
  FDRE \bytes_occ_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\out_tkeep[7]_INST_0_i_1_n_0 ),
        .Q(bytes_occ[0]),
        .R(1'b0));
  FDRE \bytes_occ_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\out_tkeep[7]_INST_0_i_3_n_0 ),
        .Q(bytes_occ[1]),
        .R(1'b0));
  FDRE \bytes_occ_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\out_tkeep[7]_INST_0_i_2_n_0 ),
        .Q(bytes_occ[2]),
        .R(1'b0));
  LUT5 #(
    .INIT(32'h4540EFEF)) 
    \data_shift[0]_i_1 
       (.I0(state__0[2]),
        .I1(\data_shift[0]_i_2_n_0 ),
        .I2(state__0[1]),
        .I3(state__0[0]),
        .I4(\data_shift[0]_i_3_n_0 ),
        .O(\data_shift[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT4 #(
    .INIT(16'h2F20)) 
    \data_shift[0]_i_2 
       (.I0(data_shift_next[2]),
        .I1(data_shift_next[1]),
        .I2(state__0[0]),
        .I3(\data_shift[2]_i_2_n_0 ),
        .O(\data_shift[0]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT4 #(
    .INIT(16'hFFFB)) 
    \data_shift[0]_i_3 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[4]),
        .I2(data_shift_next[3]),
        .I3(data_shift_next[2]),
        .O(\data_shift[0]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hEFEF4540)) 
    \data_shift[1]_i_1 
       (.I0(state__0[2]),
        .I1(\data_shift[1]_i_2_n_0 ),
        .I2(state__0[1]),
        .I3(state__0[0]),
        .I4(in8[1]),
        .O(\data_shift[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT4 #(
    .INIT(16'hFBF8)) 
    \data_shift[1]_i_2 
       (.I0(data_shift_next[2]),
        .I1(state__0[0]),
        .I2(data_shift_next[1]),
        .I3(\data_shift[2]_i_2_n_0 ),
        .O(\data_shift[1]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT4 #(
    .INIT(16'hFF10)) 
    \data_shift[1]_i_3 
       (.I0(data_shift_next[2]),
        .I1(data_shift_next[3]),
        .I2(data_shift_next[4]),
        .I3(data_shift_next[1]),
        .O(in8[1]));
  LUT6 #(
    .INIT(64'hFFDCFFFF00DC00AA)) 
    \data_shift[2]_i_1 
       (.I0(state__0[0]),
        .I1(data_shift_next[2]),
        .I2(\data_shift[2]_i_2_n_0 ),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(in8[2]),
        .O(\data_shift[2]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h00000001)) 
    \data_shift[2]_i_2 
       (.I0(data_shift_next[6]),
        .I1(data_shift_next[5]),
        .I2(data_shift_next[7]),
        .I3(data_shift_next[1]),
        .I4(\data_shift[2]_i_4_n_0 ),
        .O(\data_shift[2]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT4 #(
    .INIT(16'hAABA)) 
    \data_shift[2]_i_3 
       (.I0(data_shift_next[2]),
        .I1(data_shift_next[3]),
        .I2(data_shift_next[4]),
        .I3(data_shift_next[1]),
        .O(in8[2]));
  LUT4 #(
    .INIT(16'hFFEF)) 
    \data_shift[2]_i_4 
       (.I0(data_shift_next[3]),
        .I1(data_shift_next[4]),
        .I2(\data_shift_reg_n_0_[7] ),
        .I3(data_shift_next[2]),
        .O(\data_shift[2]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hEFEF4540)) 
    \data_shift[3]_i_1 
       (.I0(state__0[2]),
        .I1(\data_shift[3]_i_2_n_0 ),
        .I2(state__0[1]),
        .I3(state__0[0]),
        .I4(in8[3]),
        .O(\data_shift[3]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF4FFF40)) 
    \data_shift[3]_i_2 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[2]),
        .I2(state__0[0]),
        .I3(data_shift_next[3]),
        .I4(\data_shift[2]_i_2_n_0 ),
        .O(\data_shift[3]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT4 #(
    .INIT(16'hCCDC)) 
    \data_shift[3]_i_3 
       (.I0(data_shift_next[2]),
        .I1(data_shift_next[3]),
        .I2(data_shift_next[4]),
        .I3(data_shift_next[1]),
        .O(in8[3]));
  LUT5 #(
    .INIT(32'hB8BBB8AA)) 
    \data_shift[4]_i_1 
       (.I0(data_shift_next[4]),
        .I1(state__0[2]),
        .I2(\data_shift[4]_i_2_n_0 ),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .O(\data_shift[4]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF4FFF40)) 
    \data_shift[4]_i_2 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[2]),
        .I2(state__0[0]),
        .I3(data_shift_next[4]),
        .I4(\data_shift[2]_i_2_n_0 ),
        .O(\data_shift[4]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hEFEF4540)) 
    \data_shift[5]_i_1 
       (.I0(state__0[2]),
        .I1(\data_shift[5]_i_2_n_0 ),
        .I2(state__0[1]),
        .I3(state__0[0]),
        .I4(in8[5]),
        .O(\data_shift[5]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFF4FFF40)) 
    \data_shift[5]_i_2 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[2]),
        .I2(state__0[0]),
        .I3(data_shift_next[5]),
        .I4(\data_shift[2]_i_2_n_0 ),
        .O(\data_shift[5]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT5 #(
    .INIT(32'hAAAAABAA)) 
    \data_shift[5]_i_3 
       (.I0(data_shift_next[5]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[3]),
        .I3(data_shift_next[4]),
        .I4(data_shift_next[1]),
        .O(in8[5]));
  LUT5 #(
    .INIT(32'hEFEF4540)) 
    \data_shift[6]_i_1 
       (.I0(state__0[2]),
        .I1(\data_shift[6]_i_2_n_0 ),
        .I2(state__0[1]),
        .I3(state__0[0]),
        .I4(in8[6]),
        .O(\data_shift[6]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'hFF4FFF40)) 
    \data_shift[6]_i_2 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[2]),
        .I2(state__0[0]),
        .I3(data_shift_next[6]),
        .I4(\data_shift[2]_i_2_n_0 ),
        .O(\data_shift[6]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT5 #(
    .INIT(32'hAAAAABAA)) 
    \data_shift[6]_i_3 
       (.I0(data_shift_next[6]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[3]),
        .I3(data_shift_next[4]),
        .I4(data_shift_next[1]),
        .O(in8[6]));
  LUT5 #(
    .INIT(32'h33001F00)) 
    \data_shift[7]_i_1 
       (.I0(\hdr_shift_reg_n_0_[2] ),
        .I1(state__0[2]),
        .I2(state__0[0]),
        .I3(p_2_in15_out),
        .I4(state__0[1]),
        .O(data_shift));
  LUT5 #(
    .INIT(32'hEFEF4540)) 
    \data_shift[7]_i_2 
       (.I0(state__0[2]),
        .I1(\data_shift[7]_i_3_n_0 ),
        .I2(state__0[1]),
        .I3(state__0[0]),
        .I4(in8[7]),
        .O(\data_shift[7]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'hFF4FFF40)) 
    \data_shift[7]_i_3 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[2]),
        .I2(state__0[0]),
        .I3(data_shift_next[7]),
        .I4(\data_shift[2]_i_2_n_0 ),
        .O(\data_shift[7]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT5 #(
    .INIT(32'hAAAAABAA)) 
    \data_shift[7]_i_4 
       (.I0(data_shift_next[7]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[3]),
        .I3(data_shift_next[4]),
        .I4(data_shift_next[1]),
        .O(in8[7]));
  FDSE \data_shift_reg[0] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[0]_i_1_n_0 ),
        .Q(data_shift_next[1]),
        .S(rst));
  FDSE \data_shift_reg[1] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[1]_i_1_n_0 ),
        .Q(data_shift_next[2]),
        .S(rst));
  FDSE \data_shift_reg[2] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[2]_i_1_n_0 ),
        .Q(data_shift_next[3]),
        .S(rst));
  FDSE \data_shift_reg[3] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[3]_i_1_n_0 ),
        .Q(data_shift_next[4]),
        .S(rst));
  FDSE \data_shift_reg[4] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[4]_i_1_n_0 ),
        .Q(data_shift_next[5]),
        .S(rst));
  FDSE \data_shift_reg[5] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[5]_i_1_n_0 ),
        .Q(data_shift_next[6]),
        .S(rst));
  FDSE \data_shift_reg[6] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[6]_i_1_n_0 ),
        .Q(data_shift_next[7]),
        .S(rst));
  FDSE \data_shift_reg[7] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[7]_i_2_n_0 ),
        .Q(\data_shift_reg_n_0_[7] ),
        .S(rst));
  LUT6 #(
    .INIT(64'hFFFFFFF800000008)) 
    extra_cycle_i_1
       (.I0(p_2_in15_out),
        .I1(in_tlast),
        .I2(extra_cycle_i_2_n_0),
        .I3(state__0[0]),
        .I4(rst),
        .I5(extra_cycle_reg_n_0),
        .O(extra_cycle_i_1_n_0));
  LUT6 #(
    .INIT(64'hFFFF8AAA8AAAFFFF)) 
    extra_cycle_i_2
       (.I0(extra_cycle_i_3_n_0),
        .I1(out_tlast),
        .I2(in_tlast),
        .I3(p_2_in15_out),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(extra_cycle_i_2_n_0));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT5 #(
    .INIT(32'h7F7F77FF)) 
    extra_cycle_i_3
       (.I0(extra_cycle_reg_n_0),
        .I1(out_tready),
        .I2(in_tvalid),
        .I3(out_tlast),
        .I4(out_tvalid_i1),
        .O(extra_cycle_i_3_n_0));
  FDRE #(
    .INIT(1'b0)) 
    extra_cycle_reg
       (.C(clk),
        .CE(1'b1),
        .D(extra_cycle_i_1_n_0),
        .Q(extra_cycle_reg_n_0),
        .R(1'b0));
  LUT6 #(
    .INIT(64'hFFFFFFFFA9AAAAAA)) 
    \hdr_shift[2]_i_1 
       (.I0(\hdr_shift_reg_n_0_[2] ),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .I3(state__0[0]),
        .I4(p_2_in15_out),
        .I5(rst),
        .O(\hdr_shift[2]_i_1_n_0 ));
  FDRE \hdr_shift_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\hdr_shift[2]_i_1_n_0 ),
        .Q(\hdr_shift_reg_n_0_[2] ),
        .R(1'b0));
  LUT2 #(
    .INIT(4'h2)) 
    in_tready_INST_0
       (.I0(out_tready),
        .I1(extra_cycle_reg_n_0),
        .O(in_tready));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[0]_INST_0 
       (.I0(in_tdata[0]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[0]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[0]_INST_0_i_2_n_0 ),
        .O(out_tdata[0]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[0]_INST_0_i_1 
       (.I0(in_tdata[0]),
        .I1(state__0[0]),
        .I2(p_1_in[0]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[0]),
        .O(\out_tdata[0]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[0]_INST_0_i_2 
       (.I0(\out_tdata[0]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[0]_INST_0_i_5_n_0 ),
        .O(\out_tdata[0]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[0]_INST_0_i_3 
       (.I0(in_tdata[4]),
        .I1(p_1_in[0]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[0]),
        .I4(data_shift_next[2]),
        .I5(data3[0]),
        .O(p_2_in__0[0]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[0]_INST_0_i_4 
       (.I0(in_tdata[4]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[0]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[0]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[0]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[0]),
        .I2(state__0[0]),
        .I3(p_1_in__0[0]),
        .I4(\out_tdata[0]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[0]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[0]_INST_0_i_6 
       (.I0(\out_tdata[0]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[0]_INST_0_i_8_n_0 ),
        .O(\out_tdata[0]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[0]_INST_0_i_7 
       (.I0(data3[0]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[0]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[4]),
        .O(\out_tdata[0]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[0]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(\samp24_last_reg_n_0_[8] ),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[0]),
        .O(\out_tdata[0]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[10]_INST_0 
       (.I0(in_tdata[10]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[10]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[10]_INST_0_i_2_n_0 ),
        .O(out_tdata[10]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[10]_INST_0_i_1 
       (.I0(in_tdata[10]),
        .I1(state__0[0]),
        .I2(p_1_in[10]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[10]),
        .O(\out_tdata[10]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[10]_INST_0_i_2 
       (.I0(\out_tdata[10]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[10]_INST_0_i_5_n_0 ),
        .O(\out_tdata[10]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[10]_INST_0_i_3 
       (.I0(in_tdata[14]),
        .I1(p_1_in[10]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[10]),
        .I4(data_shift_next[2]),
        .I5(\samp48_last_reg_n_0_[26] ),
        .O(p_2_in__0[10]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[10]_INST_0_i_4 
       (.I0(in_tdata[22]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[10]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[10]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[10]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[10]),
        .I2(state__0[0]),
        .I3(p_1_in__0[10]),
        .I4(\out_tdata[10]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[10]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[10]_INST_0_i_6 
       (.I0(\out_tdata[10]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[10]_INST_0_i_8_n_0 ),
        .O(\out_tdata[10]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[10]_INST_0_i_7 
       (.I0(in_tdata[6]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[10]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[14]),
        .O(\out_tdata[10]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[10]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(data3[2]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[10]),
        .O(\out_tdata[10]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[11]_INST_0 
       (.I0(in_tdata[11]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[11]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[11]_INST_0_i_2_n_0 ),
        .O(out_tdata[11]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[11]_INST_0_i_1 
       (.I0(in_tdata[11]),
        .I1(state__0[0]),
        .I2(p_1_in[11]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[11]),
        .O(\out_tdata[11]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[11]_INST_0_i_2 
       (.I0(\out_tdata[11]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[11]_INST_0_i_5_n_0 ),
        .O(\out_tdata[11]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[11]_INST_0_i_3 
       (.I0(in_tdata[15]),
        .I1(p_1_in[11]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[11]),
        .I4(data_shift_next[2]),
        .I5(\samp48_last_reg_n_0_[27] ),
        .O(p_2_in__0[11]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[11]_INST_0_i_4 
       (.I0(in_tdata[23]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[11]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[11]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[11]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[11]),
        .I2(state__0[0]),
        .I3(p_1_in__0[11]),
        .I4(\out_tdata[11]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[11]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[11]_INST_0_i_6 
       (.I0(\out_tdata[11]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[11]_INST_0_i_8_n_0 ),
        .O(\out_tdata[11]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[11]_INST_0_i_7 
       (.I0(in_tdata[7]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[11]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[15]),
        .O(\out_tdata[11]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[11]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(data3[3]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[11]),
        .O(\out_tdata[11]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[12]_INST_0 
       (.I0(in_tdata[12]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[12]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[12]_INST_0_i_2_n_0 ),
        .O(out_tdata[12]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[12]_INST_0_i_1 
       (.I0(in_tdata[12]),
        .I1(state__0[0]),
        .I2(p_1_in[12]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[12]),
        .O(\out_tdata[12]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[12]_INST_0_i_2 
       (.I0(\out_tdata[12]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[12]_INST_0_i_5_n_0 ),
        .O(\out_tdata[12]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[12]_INST_0_i_3 
       (.I0(in_tdata[20]),
        .I1(p_1_in[12]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[12]),
        .I4(data_shift_next[2]),
        .I5(\samp48_last_reg_n_0_[28] ),
        .O(p_2_in__0[12]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[12]_INST_0_i_4 
       (.I0(in_tdata[24]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[12]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[12]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[12]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[12]),
        .I2(state__0[0]),
        .I3(p_1_in__0[12]),
        .I4(\out_tdata[12]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[12]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[12]_INST_0_i_6 
       (.I0(\out_tdata[12]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[12]_INST_0_i_8_n_0 ),
        .O(\out_tdata[12]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[12]_INST_0_i_7 
       (.I0(in_tdata[8]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[12]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[20]),
        .O(\out_tdata[12]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[12]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(data3[4]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[12]),
        .O(\out_tdata[12]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[13]_INST_0 
       (.I0(in_tdata[13]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[13]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[13]_INST_0_i_2_n_0 ),
        .O(out_tdata[13]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[13]_INST_0_i_1 
       (.I0(in_tdata[13]),
        .I1(state__0[0]),
        .I2(p_1_in[13]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[13]),
        .O(\out_tdata[13]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[13]_INST_0_i_2 
       (.I0(\out_tdata[13]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[13]_INST_0_i_5_n_0 ),
        .O(\out_tdata[13]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[13]_INST_0_i_3 
       (.I0(in_tdata[21]),
        .I1(p_1_in[13]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[13]),
        .I4(data_shift_next[2]),
        .I5(\samp48_last_reg_n_0_[29] ),
        .O(p_2_in__0[13]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[13]_INST_0_i_4 
       (.I0(in_tdata[25]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[13]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[13]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[13]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[13]),
        .I2(state__0[0]),
        .I3(p_1_in__0[13]),
        .I4(\out_tdata[13]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[13]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[13]_INST_0_i_6 
       (.I0(\out_tdata[13]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[13]_INST_0_i_8_n_0 ),
        .O(\out_tdata[13]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[13]_INST_0_i_7 
       (.I0(in_tdata[9]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[13]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[21]),
        .O(\out_tdata[13]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[13]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(data3[5]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[13]),
        .O(\out_tdata[13]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[14]_INST_0 
       (.I0(in_tdata[14]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[14]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[14]_INST_0_i_2_n_0 ),
        .O(out_tdata[14]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[14]_INST_0_i_1 
       (.I0(in_tdata[14]),
        .I1(state__0[0]),
        .I2(p_1_in[14]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[14]),
        .O(\out_tdata[14]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[14]_INST_0_i_2 
       (.I0(\out_tdata[14]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[14]_INST_0_i_5_n_0 ),
        .O(\out_tdata[14]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[14]_INST_0_i_3 
       (.I0(in_tdata[22]),
        .I1(p_1_in[14]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[14]),
        .I4(data_shift_next[2]),
        .I5(\samp48_last_reg_n_0_[30] ),
        .O(p_2_in__0[14]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[14]_INST_0_i_4 
       (.I0(in_tdata[26]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[14]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[14]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[14]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[14]),
        .I2(state__0[0]),
        .I3(p_1_in__0[14]),
        .I4(\out_tdata[14]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[14]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[14]_INST_0_i_6 
       (.I0(\out_tdata[14]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[14]_INST_0_i_8_n_0 ),
        .O(\out_tdata[14]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[14]_INST_0_i_7 
       (.I0(in_tdata[10]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[14]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[22]),
        .O(\out_tdata[14]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[14]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(data3[6]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[14]),
        .O(\out_tdata[14]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[15]_INST_0 
       (.I0(in_tdata[15]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[15]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[15]_INST_0_i_2_n_0 ),
        .O(out_tdata[15]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[15]_INST_0_i_1 
       (.I0(in_tdata[15]),
        .I1(state__0[0]),
        .I2(p_1_in[15]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[15]),
        .O(\out_tdata[15]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[15]_INST_0_i_2 
       (.I0(\out_tdata[15]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[15]_INST_0_i_5_n_0 ),
        .O(\out_tdata[15]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[15]_INST_0_i_3 
       (.I0(in_tdata[23]),
        .I1(p_1_in[15]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[15]),
        .I4(data_shift_next[2]),
        .I5(\samp48_last_reg_n_0_[31] ),
        .O(p_2_in__0[15]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[15]_INST_0_i_4 
       (.I0(in_tdata[27]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[15]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[15]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[15]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[15]),
        .I2(state__0[0]),
        .I3(p_1_in__0[15]),
        .I4(\out_tdata[15]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[15]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[15]_INST_0_i_6 
       (.I0(\out_tdata[15]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[15]_INST_0_i_8_n_0 ),
        .O(\out_tdata[15]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[15]_INST_0_i_7 
       (.I0(in_tdata[11]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[15]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[23]),
        .O(\out_tdata[15]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[15]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(data3[7]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[15]),
        .O(\out_tdata[15]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[16]_INST_0 
       (.I0(in_tdata[16]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[16]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[16]_INST_0_i_2_n_0 ),
        .O(out_tdata[16]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[16]_INST_0_i_1 
       (.I0(in_tdata[16]),
        .I1(state__0[0]),
        .I2(in_tdata[4]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[16]),
        .O(\out_tdata[16]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[16]_INST_0_i_2 
       (.I0(\out_tdata[16]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[16]_INST_0_i_5_n_0 ),
        .O(\out_tdata[16]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[16]_INST_0_i_3 
       (.I0(in_tdata[24]),
        .I1(in_tdata[4]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[16]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[0]),
        .O(p_2_in__0[16]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[16]_INST_0_i_4 
       (.I0(p_1_in__0[16]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[4]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[16]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[16]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[16]),
        .I2(state__0[0]),
        .I3(p_1_in__0[16]),
        .I4(\out_tdata[16]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[16]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[16]_INST_0_i_6 
       (.I0(\out_tdata[16]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[16]_INST_0_i_8_n_0 ),
        .O(\out_tdata[16]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[16]_INST_0_i_7 
       (.I0(in_tdata[12]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[16]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[24]),
        .O(\out_tdata[16]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[16]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[4]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[16]),
        .O(\out_tdata[16]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[17]_INST_0 
       (.I0(in_tdata[17]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[17]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[17]_INST_0_i_2_n_0 ),
        .O(out_tdata[17]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[17]_INST_0_i_1 
       (.I0(in_tdata[17]),
        .I1(state__0[0]),
        .I2(in_tdata[5]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[17]),
        .O(\out_tdata[17]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[17]_INST_0_i_2 
       (.I0(\out_tdata[17]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[17]_INST_0_i_5_n_0 ),
        .O(\out_tdata[17]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[17]_INST_0_i_3 
       (.I0(in_tdata[25]),
        .I1(in_tdata[5]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[17]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[1]),
        .O(p_2_in__0[17]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[17]_INST_0_i_4 
       (.I0(p_1_in__0[17]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[5]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[17]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[17]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[17]),
        .I2(state__0[0]),
        .I3(p_1_in__0[17]),
        .I4(\out_tdata[17]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[17]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[17]_INST_0_i_6 
       (.I0(\out_tdata[17]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[17]_INST_0_i_8_n_0 ),
        .O(\out_tdata[17]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[17]_INST_0_i_7 
       (.I0(in_tdata[13]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[17]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[25]),
        .O(\out_tdata[17]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[17]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[5]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[17]),
        .O(\out_tdata[17]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[18]_INST_0 
       (.I0(in_tdata[18]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[18]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[18]_INST_0_i_2_n_0 ),
        .O(out_tdata[18]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[18]_INST_0_i_1 
       (.I0(in_tdata[18]),
        .I1(state__0[0]),
        .I2(in_tdata[6]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[18]),
        .O(\out_tdata[18]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[18]_INST_0_i_2 
       (.I0(\out_tdata[18]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[18]_INST_0_i_5_n_0 ),
        .O(\out_tdata[18]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[18]_INST_0_i_3 
       (.I0(in_tdata[26]),
        .I1(in_tdata[6]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[18]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[2]),
        .O(p_2_in__0[18]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[18]_INST_0_i_4 
       (.I0(p_1_in__0[18]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[6]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[18]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[18]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[18]),
        .I2(state__0[0]),
        .I3(p_1_in__0[18]),
        .I4(\out_tdata[18]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[18]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[18]_INST_0_i_6 
       (.I0(\out_tdata[18]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[18]_INST_0_i_8_n_0 ),
        .O(\out_tdata[18]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[18]_INST_0_i_7 
       (.I0(in_tdata[14]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[18]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[26]),
        .O(\out_tdata[18]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[18]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[6]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[18]),
        .O(\out_tdata[18]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[19]_INST_0 
       (.I0(in_tdata[19]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(out_tdata[19]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[19]_INST_0_i_1 
       (.I0(in_tdata[19]),
        .I1(state__0[0]),
        .I2(in_tdata[7]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[19]),
        .O(\out_tdata[19]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[19]_INST_0_i_2 
       (.I0(\out_tdata[19]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[19]_INST_0_i_5_n_0 ),
        .O(\out_tdata[19]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[19]_INST_0_i_3 
       (.I0(in_tdata[27]),
        .I1(in_tdata[7]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[19]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[3]),
        .O(p_2_in__0[19]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[19]_INST_0_i_4 
       (.I0(p_1_in__0[19]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[7]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[19]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[19]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[19]),
        .I2(state__0[0]),
        .I3(p_1_in__0[19]),
        .I4(\out_tdata[19]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[19]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[19]_INST_0_i_6 
       (.I0(\out_tdata[19]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[19]_INST_0_i_8_n_0 ),
        .O(\out_tdata[19]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[19]_INST_0_i_7 
       (.I0(in_tdata[15]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[19]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[27]),
        .O(\out_tdata[19]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[19]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[7]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[19]),
        .O(\out_tdata[19]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[1]_INST_0 
       (.I0(in_tdata[1]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[1]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[1]_INST_0_i_2_n_0 ),
        .O(out_tdata[1]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[1]_INST_0_i_1 
       (.I0(in_tdata[1]),
        .I1(state__0[0]),
        .I2(p_1_in[1]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[1]),
        .O(\out_tdata[1]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[1]_INST_0_i_2 
       (.I0(\out_tdata[1]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[1]_INST_0_i_5_n_0 ),
        .O(\out_tdata[1]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[1]_INST_0_i_3 
       (.I0(in_tdata[5]),
        .I1(p_1_in[1]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[1]),
        .I4(data_shift_next[2]),
        .I5(data3[1]),
        .O(p_2_in__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[1]_INST_0_i_4 
       (.I0(in_tdata[5]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[1]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[1]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[1]),
        .I2(state__0[0]),
        .I3(p_1_in__0[1]),
        .I4(\out_tdata[1]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[1]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[1]_INST_0_i_6 
       (.I0(\out_tdata[1]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[1]_INST_0_i_8_n_0 ),
        .O(\out_tdata[1]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[1]_INST_0_i_7 
       (.I0(data3[1]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[1]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[5]),
        .O(\out_tdata[1]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[1]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(\samp24_last_reg_n_0_[9] ),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[1]),
        .O(\out_tdata[1]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[20]_INST_0 
       (.I0(in_tdata[20]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[20]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[20]_INST_0_i_2_n_0 ),
        .O(out_tdata[20]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[20]_INST_0_i_1 
       (.I0(in_tdata[20]),
        .I1(state__0[0]),
        .I2(in_tdata[8]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[20]),
        .O(\out_tdata[20]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[20]_INST_0_i_2 
       (.I0(\out_tdata[20]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[20]_INST_0_i_5_n_0 ),
        .O(\out_tdata[20]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[20]_INST_0_i_3 
       (.I0(in_tdata[28]),
        .I1(in_tdata[8]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[20]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[4]),
        .O(p_2_in__0[20]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[20]_INST_0_i_4 
       (.I0(p_1_in__0[20]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[8]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[20]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[20]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[20]),
        .I2(state__0[0]),
        .I3(p_1_in__0[20]),
        .I4(\out_tdata[20]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[20]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[20]_INST_0_i_6 
       (.I0(\out_tdata[20]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[20]_INST_0_i_8_n_0 ),
        .O(\out_tdata[20]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[20]_INST_0_i_7 
       (.I0(in_tdata[20]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[20]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[28]),
        .O(\out_tdata[20]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[20]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[8]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[20]),
        .O(\out_tdata[20]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[21]_INST_0 
       (.I0(in_tdata[21]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[21]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[21]_INST_0_i_2_n_0 ),
        .O(out_tdata[21]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[21]_INST_0_i_1 
       (.I0(in_tdata[21]),
        .I1(state__0[0]),
        .I2(in_tdata[9]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[21]),
        .O(\out_tdata[21]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[21]_INST_0_i_2 
       (.I0(\out_tdata[21]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[21]_INST_0_i_5_n_0 ),
        .O(\out_tdata[21]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[21]_INST_0_i_3 
       (.I0(in_tdata[29]),
        .I1(in_tdata[9]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[21]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[5]),
        .O(p_2_in__0[21]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[21]_INST_0_i_4 
       (.I0(p_1_in__0[21]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[9]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[21]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[21]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[21]),
        .I2(state__0[0]),
        .I3(p_1_in__0[21]),
        .I4(\out_tdata[21]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[21]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[21]_INST_0_i_6 
       (.I0(\out_tdata[21]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[21]_INST_0_i_8_n_0 ),
        .O(\out_tdata[21]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[21]_INST_0_i_7 
       (.I0(in_tdata[21]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[21]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[29]),
        .O(\out_tdata[21]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[21]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[9]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[21]),
        .O(\out_tdata[21]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[22]_INST_0 
       (.I0(in_tdata[22]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[22]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[22]_INST_0_i_2_n_0 ),
        .O(out_tdata[22]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[22]_INST_0_i_1 
       (.I0(in_tdata[22]),
        .I1(state__0[0]),
        .I2(in_tdata[10]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[22]),
        .O(\out_tdata[22]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[22]_INST_0_i_2 
       (.I0(\out_tdata[22]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[22]_INST_0_i_5_n_0 ),
        .O(\out_tdata[22]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[22]_INST_0_i_3 
       (.I0(in_tdata[30]),
        .I1(in_tdata[10]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[22]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[6]),
        .O(p_2_in__0[22]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[22]_INST_0_i_4 
       (.I0(p_1_in__0[22]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[10]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[22]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[22]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[22]),
        .I2(state__0[0]),
        .I3(p_1_in__0[22]),
        .I4(\out_tdata[22]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[22]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[22]_INST_0_i_6 
       (.I0(\out_tdata[22]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[22]_INST_0_i_8_n_0 ),
        .O(\out_tdata[22]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[22]_INST_0_i_7 
       (.I0(in_tdata[22]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[22]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[30]),
        .O(\out_tdata[22]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[22]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[10]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[22]),
        .O(\out_tdata[22]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[23]_INST_0 
       (.I0(in_tdata[23]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[23]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[23]_INST_0_i_2_n_0 ),
        .O(out_tdata[23]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[23]_INST_0_i_1 
       (.I0(in_tdata[23]),
        .I1(state__0[0]),
        .I2(in_tdata[11]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[23]),
        .O(\out_tdata[23]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[23]_INST_0_i_2 
       (.I0(\out_tdata[23]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[23]_INST_0_i_5_n_0 ),
        .O(\out_tdata[23]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[23]_INST_0_i_3 
       (.I0(in_tdata[31]),
        .I1(in_tdata[11]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[23]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[7]),
        .O(p_2_in__0[23]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[23]_INST_0_i_4 
       (.I0(p_1_in__0[23]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[11]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[23]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[23]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[23]),
        .I2(state__0[0]),
        .I3(p_1_in__0[23]),
        .I4(\out_tdata[23]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[23]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[23]_INST_0_i_6 
       (.I0(\out_tdata[23]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[23]_INST_0_i_8_n_0 ),
        .O(\out_tdata[23]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[23]_INST_0_i_7 
       (.I0(in_tdata[23]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[23]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[31]),
        .O(\out_tdata[23]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[23]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[11]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[23]),
        .O(\out_tdata[23]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[24]_INST_0 
       (.I0(in_tdata[24]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[24]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[24]_INST_0_i_2_n_0 ),
        .O(out_tdata[24]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[24]_INST_0_i_1 
       (.I0(in_tdata[24]),
        .I1(state__0[0]),
        .I2(in_tdata[12]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[24]),
        .O(\out_tdata[24]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[24]_INST_0_i_2 
       (.I0(\out_tdata[24]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[24]_INST_0_i_5_n_0 ),
        .O(\out_tdata[24]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[24]_INST_0_i_3 
       (.I0(in_tdata[36]),
        .I1(in_tdata[12]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[24]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[8]),
        .O(p_2_in__0[24]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[24]_INST_0_i_4 
       (.I0(p_1_in__0[24]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[20]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[24]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[24]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[24]),
        .I2(state__0[0]),
        .I3(p_1_in__0[24]),
        .I4(\out_tdata[24]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[24]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[24]_INST_0_i_6 
       (.I0(\out_tdata[24]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[24]_INST_0_i_8_n_0 ),
        .O(\out_tdata[24]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hAFC0A0C0)) 
    \out_tdata[24]_INST_0_i_7 
       (.I0(in_tdata[24]),
        .I1(p_1_in__0[24]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[4]),
        .O(\out_tdata[24]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[24]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[12]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[24]),
        .O(\out_tdata[24]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[25]_INST_0 
       (.I0(in_tdata[25]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[25]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[25]_INST_0_i_2_n_0 ),
        .O(out_tdata[25]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[25]_INST_0_i_1 
       (.I0(in_tdata[25]),
        .I1(state__0[0]),
        .I2(in_tdata[13]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[25]),
        .O(\out_tdata[25]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[25]_INST_0_i_2 
       (.I0(\out_tdata[25]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[25]_INST_0_i_5_n_0 ),
        .O(\out_tdata[25]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[25]_INST_0_i_3 
       (.I0(in_tdata[37]),
        .I1(in_tdata[13]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[25]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[9]),
        .O(p_2_in__0[25]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[25]_INST_0_i_4 
       (.I0(p_1_in__0[25]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[21]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[25]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[25]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[25]),
        .I2(state__0[0]),
        .I3(p_1_in__0[25]),
        .I4(\out_tdata[25]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[25]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[25]_INST_0_i_6 
       (.I0(\out_tdata[25]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[25]_INST_0_i_8_n_0 ),
        .O(\out_tdata[25]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hAFC0A0C0)) 
    \out_tdata[25]_INST_0_i_7 
       (.I0(in_tdata[25]),
        .I1(p_1_in__0[25]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[5]),
        .O(\out_tdata[25]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[25]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[13]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[25]),
        .O(\out_tdata[25]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[26]_INST_0 
       (.I0(in_tdata[26]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[26]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[26]_INST_0_i_2_n_0 ),
        .O(out_tdata[26]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[26]_INST_0_i_1 
       (.I0(in_tdata[26]),
        .I1(state__0[0]),
        .I2(in_tdata[14]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[26]),
        .O(\out_tdata[26]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[26]_INST_0_i_2 
       (.I0(\out_tdata[26]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[26]_INST_0_i_5_n_0 ),
        .O(\out_tdata[26]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[26]_INST_0_i_3 
       (.I0(in_tdata[38]),
        .I1(in_tdata[14]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[26]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[10]),
        .O(p_2_in__0[26]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[26]_INST_0_i_4 
       (.I0(p_1_in__0[26]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[22]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[26]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[26]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[26]),
        .I2(state__0[0]),
        .I3(p_1_in__0[26]),
        .I4(\out_tdata[26]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[26]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[26]_INST_0_i_6 
       (.I0(\out_tdata[26]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[26]_INST_0_i_8_n_0 ),
        .O(\out_tdata[26]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hAFC0A0C0)) 
    \out_tdata[26]_INST_0_i_7 
       (.I0(in_tdata[26]),
        .I1(p_1_in__0[26]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[6]),
        .O(\out_tdata[26]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[26]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[14]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[26]),
        .O(\out_tdata[26]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[27]_INST_0 
       (.I0(in_tdata[27]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[27]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[27]_INST_0_i_2_n_0 ),
        .O(out_tdata[27]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[27]_INST_0_i_1 
       (.I0(in_tdata[27]),
        .I1(state__0[0]),
        .I2(in_tdata[15]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[27]),
        .O(\out_tdata[27]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[27]_INST_0_i_2 
       (.I0(\out_tdata[27]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[27]_INST_0_i_5_n_0 ),
        .O(\out_tdata[27]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[27]_INST_0_i_3 
       (.I0(in_tdata[39]),
        .I1(in_tdata[15]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[27]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[11]),
        .O(p_2_in__0[27]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[27]_INST_0_i_4 
       (.I0(p_1_in__0[27]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[23]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[27]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[27]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[27]),
        .I2(state__0[0]),
        .I3(p_1_in__0[27]),
        .I4(\out_tdata[27]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[27]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[27]_INST_0_i_6 
       (.I0(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[27]_INST_0_i_8_n_0 ),
        .O(\out_tdata[27]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hAFC0A0C0)) 
    \out_tdata[27]_INST_0_i_7 
       (.I0(in_tdata[27]),
        .I1(p_1_in__0[27]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[7]),
        .O(\out_tdata[27]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[27]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[15]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[27]),
        .O(\out_tdata[27]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[28]_INST_0 
       (.I0(in_tdata[28]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[28]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[28]_INST_0_i_2_n_0 ),
        .O(out_tdata[28]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[28]_INST_0_i_1 
       (.I0(in_tdata[28]),
        .I1(state__0[0]),
        .I2(in_tdata[20]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[28]),
        .O(\out_tdata[28]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[28]_INST_0_i_2 
       (.I0(\out_tdata[28]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[28]_INST_0_i_5_n_0 ),
        .O(\out_tdata[28]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[28]_INST_0_i_3 
       (.I0(in_tdata[40]),
        .I1(in_tdata[20]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[28]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[12]),
        .O(p_2_in__0[28]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[28]_INST_0_i_4 
       (.I0(p_1_in__0[28]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[24]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[28]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[28]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[28]),
        .I2(state__0[0]),
        .I3(p_1_in__0[28]),
        .I4(\out_tdata[28]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[28]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[28]_INST_0_i_6 
       (.I0(\out_tdata[28]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[28]_INST_0_i_8_n_0 ),
        .O(\out_tdata[28]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hAFC0A0C0)) 
    \out_tdata[28]_INST_0_i_7 
       (.I0(in_tdata[28]),
        .I1(p_1_in__0[28]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[8]),
        .O(\out_tdata[28]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[28]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[20]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[28]),
        .O(\out_tdata[28]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[29]_INST_0 
       (.I0(in_tdata[29]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[29]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[29]_INST_0_i_2_n_0 ),
        .O(out_tdata[29]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[29]_INST_0_i_1 
       (.I0(in_tdata[29]),
        .I1(state__0[0]),
        .I2(in_tdata[21]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[29]),
        .O(\out_tdata[29]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[29]_INST_0_i_2 
       (.I0(\out_tdata[29]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[29]_INST_0_i_5_n_0 ),
        .O(\out_tdata[29]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[29]_INST_0_i_3 
       (.I0(in_tdata[41]),
        .I1(in_tdata[21]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[29]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[13]),
        .O(p_2_in__0[29]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[29]_INST_0_i_4 
       (.I0(p_1_in__0[29]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[25]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[29]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[29]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[29]),
        .I2(state__0[0]),
        .I3(p_1_in__0[29]),
        .I4(\out_tdata[29]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[29]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[29]_INST_0_i_6 
       (.I0(\out_tdata[29]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[29]_INST_0_i_8_n_0 ),
        .O(\out_tdata[29]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hAFC0A0C0)) 
    \out_tdata[29]_INST_0_i_7 
       (.I0(in_tdata[29]),
        .I1(p_1_in__0[29]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[9]),
        .O(\out_tdata[29]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[29]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[21]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[29]),
        .O(\out_tdata[29]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[2]_INST_0 
       (.I0(in_tdata[2]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[2]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[2]_INST_0_i_2_n_0 ),
        .O(out_tdata[2]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[2]_INST_0_i_1 
       (.I0(in_tdata[2]),
        .I1(state__0[0]),
        .I2(p_1_in[2]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[2]),
        .O(\out_tdata[2]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[2]_INST_0_i_2 
       (.I0(\out_tdata[2]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[2]_INST_0_i_5_n_0 ),
        .O(\out_tdata[2]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[2]_INST_0_i_3 
       (.I0(in_tdata[6]),
        .I1(p_1_in[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[2]),
        .I4(data_shift_next[2]),
        .I5(data3[2]),
        .O(p_2_in__0[2]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[2]_INST_0_i_4 
       (.I0(in_tdata[6]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[2]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[2]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[2]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[2]),
        .I2(state__0[0]),
        .I3(p_1_in__0[2]),
        .I4(\out_tdata[2]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[2]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[2]_INST_0_i_6 
       (.I0(\out_tdata[2]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[2]_INST_0_i_8_n_0 ),
        .O(\out_tdata[2]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[2]_INST_0_i_7 
       (.I0(data3[2]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[2]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[6]),
        .O(\out_tdata[2]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[2]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(\samp24_last_reg_n_0_[10] ),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[2]),
        .O(\out_tdata[2]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[30]_INST_0 
       (.I0(in_tdata[30]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[30]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[30]_INST_0_i_2_n_0 ),
        .O(out_tdata[30]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[30]_INST_0_i_1 
       (.I0(in_tdata[30]),
        .I1(state__0[0]),
        .I2(in_tdata[22]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[30]),
        .O(\out_tdata[30]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[30]_INST_0_i_2 
       (.I0(\out_tdata[30]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[30]_INST_0_i_5_n_0 ),
        .O(\out_tdata[30]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[30]_INST_0_i_3 
       (.I0(in_tdata[42]),
        .I1(in_tdata[22]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[30]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[14]),
        .O(p_2_in__0[30]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[30]_INST_0_i_4 
       (.I0(p_1_in__0[30]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[26]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[30]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[30]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[30]),
        .I2(state__0[0]),
        .I3(p_1_in__0[30]),
        .I4(\out_tdata[30]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[30]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[30]_INST_0_i_6 
       (.I0(\out_tdata[30]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[30]_INST_0_i_8_n_0 ),
        .O(\out_tdata[30]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hAFC0A0C0)) 
    \out_tdata[30]_INST_0_i_7 
       (.I0(in_tdata[30]),
        .I1(p_1_in__0[30]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[10]),
        .O(\out_tdata[30]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[30]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[22]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[30]),
        .O(\out_tdata[30]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[31]_INST_0 
       (.I0(in_tdata[31]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[31]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[31]_INST_0_i_2_n_0 ),
        .O(out_tdata[31]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[31]_INST_0_i_1 
       (.I0(in_tdata[31]),
        .I1(state__0[0]),
        .I2(in_tdata[23]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[31]),
        .O(\out_tdata[31]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[31]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[31]_INST_0_i_5_n_0 ),
        .O(\out_tdata[31]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[31]_INST_0_i_3 
       (.I0(in_tdata[43]),
        .I1(in_tdata[23]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[31]),
        .I4(data_shift_next[2]),
        .I5(p_1_in[15]),
        .O(p_2_in__0[31]));
  LUT6 #(
    .INIT(64'h00000000AAAA2E22)) 
    \out_tdata[31]_INST_0_i_4 
       (.I0(p_1_in__0[31]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[27]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[31]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[31]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[31]),
        .I2(state__0[0]),
        .I3(p_1_in__0[31]),
        .I4(\out_tdata[31]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[31]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[31]_INST_0_i_6 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[31]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hAFC0A0C0)) 
    \out_tdata[31]_INST_0_i_7 
       (.I0(in_tdata[31]),
        .I1(p_1_in__0[31]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[11]),
        .O(\out_tdata[31]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[31]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(in_tdata[23]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[31]),
        .O(\out_tdata[31]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[32]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[32]),
        .I2(state__0[0]),
        .I3(in14[32]),
        .I4(state__0[2]),
        .I5(\out_tdata[32]_INST_0_i_2_n_0 ),
        .O(out_tdata[32]));
  LUT6 #(
    .INIT(64'hEF40FF55EF40AA00)) 
    \out_tdata[32]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[44]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[24]),
        .I4(data_shift_next[1]),
        .I5(\out_tdata[32]_INST_0_i_3_n_0 ),
        .O(in14[32]));
  MUXF7 \out_tdata[32]_INST_0_i_2 
       (.I0(\out_tdata[32]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[32]_INST_0_i_5_n_0 ),
        .O(\out_tdata[32]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[32]_INST_0_i_3 
       (.I0(p_1_in__0[32]),
        .I1(data_shift_next[2]),
        .I2(in_tdata[4]),
        .O(\out_tdata[32]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[32]_INST_0_i_4 
       (.I0(in_tdata[4]),
        .I1(data_shift_next[2]),
        .I2(p_1_in__0[32]),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[32]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[32]_INST_0_i_5 
       (.I0(samp32[0]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(p_1_in__0[32]),
        .I4(\out_tdata[32]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[32]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[32]_INST_0_i_6 
       (.I0(in_tdata[4]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[0]),
        .O(samp32[0]));
  MUXF7 \out_tdata[32]_INST_0_i_7 
       (.I0(\out_tdata[32]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[32]_INST_0_i_9_n_0 ),
        .O(\out_tdata[32]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[32]_INST_0_i_8 
       (.I0(p_1_in__0[32]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[12]),
        .O(\out_tdata[32]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \out_tdata[32]_INST_0_i_9 
       (.I0(in_tdata[24]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[32]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[4]),
        .O(\out_tdata[32]_INST_0_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[33]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[33]),
        .I2(state__0[0]),
        .I3(in14[33]),
        .I4(state__0[2]),
        .I5(\out_tdata[33]_INST_0_i_2_n_0 ),
        .O(out_tdata[33]));
  LUT6 #(
    .INIT(64'hEF40FF55EF40AA00)) 
    \out_tdata[33]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[45]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[25]),
        .I4(data_shift_next[1]),
        .I5(\out_tdata[33]_INST_0_i_3_n_0 ),
        .O(in14[33]));
  MUXF7 \out_tdata[33]_INST_0_i_2 
       (.I0(\out_tdata[33]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[33]_INST_0_i_5_n_0 ),
        .O(\out_tdata[33]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[33]_INST_0_i_3 
       (.I0(p_1_in__0[33]),
        .I1(data_shift_next[2]),
        .I2(in_tdata[5]),
        .O(\out_tdata[33]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[33]_INST_0_i_4 
       (.I0(in_tdata[5]),
        .I1(data_shift_next[2]),
        .I2(p_1_in__0[33]),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[33]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[33]_INST_0_i_5 
       (.I0(samp32[1]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(p_1_in__0[33]),
        .I4(\out_tdata[33]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[33]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[33]_INST_0_i_6 
       (.I0(in_tdata[5]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[1]),
        .O(samp32[1]));
  MUXF7 \out_tdata[33]_INST_0_i_7 
       (.I0(\out_tdata[33]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[33]_INST_0_i_9_n_0 ),
        .O(\out_tdata[33]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[33]_INST_0_i_8 
       (.I0(p_1_in__0[33]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[13]),
        .O(\out_tdata[33]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \out_tdata[33]_INST_0_i_9 
       (.I0(in_tdata[25]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[33]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[5]),
        .O(\out_tdata[33]_INST_0_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[34]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[34]),
        .I2(state__0[0]),
        .I3(in14[34]),
        .I4(state__0[2]),
        .I5(\out_tdata[34]_INST_0_i_2_n_0 ),
        .O(out_tdata[34]));
  LUT6 #(
    .INIT(64'hEF40FF55EF40AA00)) 
    \out_tdata[34]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[46]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[26]),
        .I4(data_shift_next[1]),
        .I5(\out_tdata[34]_INST_0_i_3_n_0 ),
        .O(in14[34]));
  MUXF7 \out_tdata[34]_INST_0_i_2 
       (.I0(\out_tdata[34]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[34]_INST_0_i_5_n_0 ),
        .O(\out_tdata[34]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[34]_INST_0_i_3 
       (.I0(p_1_in__0[34]),
        .I1(data_shift_next[2]),
        .I2(in_tdata[6]),
        .O(\out_tdata[34]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[34]_INST_0_i_4 
       (.I0(in_tdata[6]),
        .I1(data_shift_next[2]),
        .I2(p_1_in__0[34]),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[34]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[34]_INST_0_i_5 
       (.I0(samp32[2]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(p_1_in__0[34]),
        .I4(\out_tdata[34]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[34]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[34]_INST_0_i_6 
       (.I0(in_tdata[6]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[2]),
        .O(samp32[2]));
  MUXF7 \out_tdata[34]_INST_0_i_7 
       (.I0(\out_tdata[34]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[34]_INST_0_i_9_n_0 ),
        .O(\out_tdata[34]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[34]_INST_0_i_8 
       (.I0(p_1_in__0[34]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[14]),
        .O(\out_tdata[34]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \out_tdata[34]_INST_0_i_9 
       (.I0(in_tdata[26]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[34]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[6]),
        .O(\out_tdata[34]_INST_0_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[35]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[35]),
        .I2(state__0[0]),
        .I3(in14[35]),
        .I4(state__0[2]),
        .I5(\out_tdata[35]_INST_0_i_2_n_0 ),
        .O(out_tdata[35]));
  LUT6 #(
    .INIT(64'hEF40FF55EF40AA00)) 
    \out_tdata[35]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[47]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[27]),
        .I4(data_shift_next[1]),
        .I5(\out_tdata[35]_INST_0_i_3_n_0 ),
        .O(in14[35]));
  MUXF7 \out_tdata[35]_INST_0_i_2 
       (.I0(\out_tdata[35]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[35]_INST_0_i_5_n_0 ),
        .O(\out_tdata[35]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[35]_INST_0_i_3 
       (.I0(p_1_in__0[35]),
        .I1(data_shift_next[2]),
        .I2(in_tdata[7]),
        .O(\out_tdata[35]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[35]_INST_0_i_4 
       (.I0(in_tdata[7]),
        .I1(data_shift_next[2]),
        .I2(p_1_in__0[35]),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[35]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[35]_INST_0_i_5 
       (.I0(samp32[3]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(p_1_in__0[35]),
        .I4(\out_tdata[35]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[35]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[35]_INST_0_i_6 
       (.I0(in_tdata[7]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[3]),
        .O(samp32[3]));
  MUXF7 \out_tdata[35]_INST_0_i_7 
       (.I0(\out_tdata[35]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[35]_INST_0_i_9_n_0 ),
        .O(\out_tdata[35]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[35]_INST_0_i_8 
       (.I0(p_1_in__0[35]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[15]),
        .O(\out_tdata[35]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \out_tdata[35]_INST_0_i_9 
       (.I0(in_tdata[27]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[35]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[7]),
        .O(\out_tdata[35]_INST_0_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[36]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[36]),
        .I2(state__0[0]),
        .I3(in14[36]),
        .I4(state__0[2]),
        .I5(\out_tdata[36]_INST_0_i_2_n_0 ),
        .O(out_tdata[36]));
  LUT6 #(
    .INIT(64'hEF40FF55EF40AA00)) 
    \out_tdata[36]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[52]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[28]),
        .I4(data_shift_next[1]),
        .I5(\out_tdata[36]_INST_0_i_3_n_0 ),
        .O(in14[36]));
  MUXF7 \out_tdata[36]_INST_0_i_2 
       (.I0(\out_tdata[36]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[36]_INST_0_i_5_n_0 ),
        .O(\out_tdata[36]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[36]_INST_0_i_3 
       (.I0(p_1_in__0[36]),
        .I1(data_shift_next[2]),
        .I2(in_tdata[8]),
        .O(\out_tdata[36]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[36]_INST_0_i_4 
       (.I0(in_tdata[8]),
        .I1(data_shift_next[2]),
        .I2(p_1_in__0[36]),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[36]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[36]_INST_0_i_5 
       (.I0(samp32[4]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(p_1_in__0[36]),
        .I4(\out_tdata[36]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[36]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[36]_INST_0_i_6 
       (.I0(in_tdata[8]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[4]),
        .O(samp32[4]));
  MUXF7 \out_tdata[36]_INST_0_i_7 
       (.I0(\out_tdata[36]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[36]_INST_0_i_9_n_0 ),
        .O(\out_tdata[36]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[36]_INST_0_i_8 
       (.I0(p_1_in__0[36]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[20]),
        .O(\out_tdata[36]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \out_tdata[36]_INST_0_i_9 
       (.I0(in_tdata[28]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[36]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[8]),
        .O(\out_tdata[36]_INST_0_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[37]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[37]),
        .I2(state__0[0]),
        .I3(in14[37]),
        .I4(state__0[2]),
        .I5(\out_tdata[37]_INST_0_i_2_n_0 ),
        .O(out_tdata[37]));
  LUT6 #(
    .INIT(64'hEF40FF55EF40AA00)) 
    \out_tdata[37]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[53]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[29]),
        .I4(data_shift_next[1]),
        .I5(\out_tdata[37]_INST_0_i_3_n_0 ),
        .O(in14[37]));
  MUXF7 \out_tdata[37]_INST_0_i_2 
       (.I0(\out_tdata[37]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[37]_INST_0_i_5_n_0 ),
        .O(\out_tdata[37]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[37]_INST_0_i_3 
       (.I0(p_1_in__0[37]),
        .I1(data_shift_next[2]),
        .I2(in_tdata[9]),
        .O(\out_tdata[37]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[37]_INST_0_i_4 
       (.I0(in_tdata[9]),
        .I1(data_shift_next[2]),
        .I2(p_1_in__0[37]),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[37]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[37]_INST_0_i_5 
       (.I0(samp32[5]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(p_1_in__0[37]),
        .I4(\out_tdata[37]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[37]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[37]_INST_0_i_6 
       (.I0(in_tdata[9]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[5]),
        .O(samp32[5]));
  MUXF7 \out_tdata[37]_INST_0_i_7 
       (.I0(\out_tdata[37]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[37]_INST_0_i_9_n_0 ),
        .O(\out_tdata[37]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[37]_INST_0_i_8 
       (.I0(p_1_in__0[37]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[21]),
        .O(\out_tdata[37]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \out_tdata[37]_INST_0_i_9 
       (.I0(in_tdata[29]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[37]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[9]),
        .O(\out_tdata[37]_INST_0_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[38]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[38]),
        .I2(state__0[0]),
        .I3(in14[38]),
        .I4(state__0[2]),
        .I5(\out_tdata[38]_INST_0_i_2_n_0 ),
        .O(out_tdata[38]));
  LUT6 #(
    .INIT(64'hEF40FF55EF40AA00)) 
    \out_tdata[38]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[54]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[30]),
        .I4(data_shift_next[1]),
        .I5(\out_tdata[38]_INST_0_i_3_n_0 ),
        .O(in14[38]));
  MUXF7 \out_tdata[38]_INST_0_i_2 
       (.I0(\out_tdata[38]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[38]_INST_0_i_5_n_0 ),
        .O(\out_tdata[38]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[38]_INST_0_i_3 
       (.I0(p_1_in__0[38]),
        .I1(data_shift_next[2]),
        .I2(in_tdata[10]),
        .O(\out_tdata[38]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[38]_INST_0_i_4 
       (.I0(in_tdata[10]),
        .I1(data_shift_next[2]),
        .I2(p_1_in__0[38]),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[38]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[38]_INST_0_i_5 
       (.I0(samp32[6]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(p_1_in__0[38]),
        .I4(\out_tdata[38]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[38]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[38]_INST_0_i_6 
       (.I0(in_tdata[10]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[6]),
        .O(samp32[6]));
  MUXF7 \out_tdata[38]_INST_0_i_7 
       (.I0(\out_tdata[38]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[38]_INST_0_i_9_n_0 ),
        .O(\out_tdata[38]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[38]_INST_0_i_8 
       (.I0(p_1_in__0[38]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[22]),
        .O(\out_tdata[38]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \out_tdata[38]_INST_0_i_9 
       (.I0(in_tdata[30]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[38]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[10]),
        .O(\out_tdata[38]_INST_0_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[39]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[39]),
        .I2(state__0[0]),
        .I3(in14[39]),
        .I4(state__0[2]),
        .I5(\out_tdata[39]_INST_0_i_2_n_0 ),
        .O(out_tdata[39]));
  LUT6 #(
    .INIT(64'hEF40FF55EF40AA00)) 
    \out_tdata[39]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[55]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[31]),
        .I4(data_shift_next[1]),
        .I5(\out_tdata[39]_INST_0_i_3_n_0 ),
        .O(in14[39]));
  MUXF7 \out_tdata[39]_INST_0_i_2 
       (.I0(\out_tdata[39]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[39]_INST_0_i_5_n_0 ),
        .O(\out_tdata[39]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[39]_INST_0_i_3 
       (.I0(p_1_in__0[39]),
        .I1(data_shift_next[2]),
        .I2(in_tdata[11]),
        .O(\out_tdata[39]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[39]_INST_0_i_4 
       (.I0(in_tdata[11]),
        .I1(data_shift_next[2]),
        .I2(p_1_in__0[39]),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[39]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[39]_INST_0_i_5 
       (.I0(samp32[7]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(p_1_in__0[39]),
        .I4(\out_tdata[39]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[39]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[39]_INST_0_i_6 
       (.I0(in_tdata[11]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[7]),
        .O(samp32[7]));
  MUXF7 \out_tdata[39]_INST_0_i_7 
       (.I0(\out_tdata[39]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[39]_INST_0_i_9_n_0 ),
        .O(\out_tdata[39]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[39]_INST_0_i_8 
       (.I0(p_1_in__0[39]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[23]),
        .O(\out_tdata[39]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hF0BBF088)) 
    \out_tdata[39]_INST_0_i_9 
       (.I0(in_tdata[31]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[39]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[11]),
        .O(\out_tdata[39]_INST_0_i_9_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[3]_INST_0 
       (.I0(in_tdata[3]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[3]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[3]_INST_0_i_2_n_0 ),
        .O(out_tdata[3]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[3]_INST_0_i_1 
       (.I0(in_tdata[3]),
        .I1(state__0[0]),
        .I2(p_1_in[3]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[3]),
        .O(\out_tdata[3]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[3]_INST_0_i_2 
       (.I0(\out_tdata[3]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[3]_INST_0_i_5_n_0 ),
        .O(\out_tdata[3]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[3]_INST_0_i_3 
       (.I0(in_tdata[7]),
        .I1(p_1_in[3]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[3]),
        .I4(data_shift_next[2]),
        .I5(data3[3]),
        .O(p_2_in__0[3]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[3]_INST_0_i_4 
       (.I0(in_tdata[7]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[3]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[3]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[3]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[3]),
        .I2(state__0[0]),
        .I3(p_1_in__0[3]),
        .I4(\out_tdata[3]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[3]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[3]_INST_0_i_6 
       (.I0(\out_tdata[3]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[3]_INST_0_i_8_n_0 ),
        .O(\out_tdata[3]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[3]_INST_0_i_7 
       (.I0(data3[3]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[3]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[7]),
        .O(\out_tdata[3]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[3]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(\samp24_last_reg_n_0_[11] ),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[3]),
        .O(\out_tdata[3]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[40]_INST_0 
       (.I0(in_tdata[40]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[40]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[40]_INST_0_i_2_n_0 ),
        .O(out_tdata[40]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[40]_INST_0_i_1 
       (.I0(in_tdata[40]),
        .I1(state__0[0]),
        .I2(in_tdata[36]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[40]),
        .O(\out_tdata[40]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[40]_INST_0_i_2 
       (.I0(\out_tdata[40]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[40]_INST_0_i_5_n_0 ),
        .O(\out_tdata[40]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[40]_INST_0_i_3 
       (.I0(in_tdata[56]),
        .I1(in_tdata[36]),
        .I2(data_shift_next[1]),
        .I3(\packed_data_reg_n_0_[40] ),
        .I4(data_shift_next[2]),
        .I5(in_tdata[12]),
        .O(p_2_in__0[40]));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[40]_INST_0_i_4 
       (.I0(in_tdata[20]),
        .I1(data_shift_next[2]),
        .I2(\packed_data_reg_n_0_[40] ),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[40]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[40]_INST_0_i_5 
       (.I0(samp32[8]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[4]),
        .I4(\out_tdata[40]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[40]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[40]_INST_0_i_6 
       (.I0(in_tdata[20]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[8]),
        .O(samp32[8]));
  MUXF7 \out_tdata[40]_INST_0_i_7 
       (.I0(\out_tdata[40]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[40]_INST_0_i_9_n_0 ),
        .O(\out_tdata[40]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[40]_INST_0_i_8 
       (.I0(\packed_data_reg_n_0_[40] ),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[24]),
        .O(\out_tdata[40]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[40]_INST_0_i_9 
       (.I0(in_tdata[4]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[40] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[12]),
        .O(\out_tdata[40]_INST_0_i_9_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[41]_INST_0 
       (.I0(in_tdata[41]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[41]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[41]_INST_0_i_2_n_0 ),
        .O(out_tdata[41]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[41]_INST_0_i_1 
       (.I0(in_tdata[41]),
        .I1(state__0[0]),
        .I2(in_tdata[37]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[41]),
        .O(\out_tdata[41]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[41]_INST_0_i_2 
       (.I0(\out_tdata[41]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[41]_INST_0_i_5_n_0 ),
        .O(\out_tdata[41]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[41]_INST_0_i_3 
       (.I0(in_tdata[57]),
        .I1(in_tdata[37]),
        .I2(data_shift_next[1]),
        .I3(\packed_data_reg_n_0_[41] ),
        .I4(data_shift_next[2]),
        .I5(in_tdata[13]),
        .O(p_2_in__0[41]));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[41]_INST_0_i_4 
       (.I0(in_tdata[21]),
        .I1(data_shift_next[2]),
        .I2(\packed_data_reg_n_0_[41] ),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[41]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[41]_INST_0_i_5 
       (.I0(samp32[9]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[5]),
        .I4(\out_tdata[41]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[41]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[41]_INST_0_i_6 
       (.I0(in_tdata[21]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[9]),
        .O(samp32[9]));
  MUXF7 \out_tdata[41]_INST_0_i_7 
       (.I0(\out_tdata[41]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[41]_INST_0_i_9_n_0 ),
        .O(\out_tdata[41]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[41]_INST_0_i_8 
       (.I0(\packed_data_reg_n_0_[41] ),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[25]),
        .O(\out_tdata[41]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[41]_INST_0_i_9 
       (.I0(in_tdata[5]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[41] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[13]),
        .O(\out_tdata[41]_INST_0_i_9_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[42]_INST_0 
       (.I0(in_tdata[42]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[42]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[42]_INST_0_i_2_n_0 ),
        .O(out_tdata[42]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[42]_INST_0_i_1 
       (.I0(in_tdata[42]),
        .I1(state__0[0]),
        .I2(in_tdata[38]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[42]),
        .O(\out_tdata[42]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[42]_INST_0_i_2 
       (.I0(\out_tdata[42]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[42]_INST_0_i_5_n_0 ),
        .O(\out_tdata[42]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[42]_INST_0_i_3 
       (.I0(in_tdata[58]),
        .I1(in_tdata[38]),
        .I2(data_shift_next[1]),
        .I3(\packed_data_reg_n_0_[42] ),
        .I4(data_shift_next[2]),
        .I5(in_tdata[14]),
        .O(p_2_in__0[42]));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[42]_INST_0_i_4 
       (.I0(in_tdata[22]),
        .I1(data_shift_next[2]),
        .I2(\packed_data_reg_n_0_[42] ),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[42]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[42]_INST_0_i_5 
       (.I0(samp32[10]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[6]),
        .I4(\out_tdata[42]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[42]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[42]_INST_0_i_6 
       (.I0(in_tdata[22]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[10]),
        .O(samp32[10]));
  MUXF7 \out_tdata[42]_INST_0_i_7 
       (.I0(\out_tdata[42]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[42]_INST_0_i_9_n_0 ),
        .O(\out_tdata[42]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[42]_INST_0_i_8 
       (.I0(\packed_data_reg_n_0_[42] ),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[26]),
        .O(\out_tdata[42]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[42]_INST_0_i_9 
       (.I0(in_tdata[6]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[42] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[14]),
        .O(\out_tdata[42]_INST_0_i_9_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[43]_INST_0 
       (.I0(in_tdata[43]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[43]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[43]_INST_0_i_2_n_0 ),
        .O(out_tdata[43]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[43]_INST_0_i_1 
       (.I0(in_tdata[43]),
        .I1(state__0[0]),
        .I2(in_tdata[39]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[43]),
        .O(\out_tdata[43]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[43]_INST_0_i_2 
       (.I0(\out_tdata[43]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[43]_INST_0_i_5_n_0 ),
        .O(\out_tdata[43]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[43]_INST_0_i_3 
       (.I0(in_tdata[59]),
        .I1(in_tdata[39]),
        .I2(data_shift_next[1]),
        .I3(\packed_data_reg_n_0_[43] ),
        .I4(data_shift_next[2]),
        .I5(in_tdata[15]),
        .O(p_2_in__0[43]));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[43]_INST_0_i_4 
       (.I0(in_tdata[23]),
        .I1(data_shift_next[2]),
        .I2(\packed_data_reg_n_0_[43] ),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[43]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[43]_INST_0_i_5 
       (.I0(samp32[11]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[7]),
        .I4(\out_tdata[43]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[43]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[43]_INST_0_i_6 
       (.I0(in_tdata[23]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[11]),
        .O(samp32[11]));
  MUXF7 \out_tdata[43]_INST_0_i_7 
       (.I0(\out_tdata[43]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[43]_INST_0_i_9_n_0 ),
        .O(\out_tdata[43]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[43]_INST_0_i_8 
       (.I0(\packed_data_reg_n_0_[43] ),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[27]),
        .O(\out_tdata[43]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[43]_INST_0_i_9 
       (.I0(in_tdata[7]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[43] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[15]),
        .O(\out_tdata[43]_INST_0_i_9_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[44]_INST_0 
       (.I0(in_tdata[44]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[44]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[44]_INST_0_i_2_n_0 ),
        .O(out_tdata[44]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[44]_INST_0_i_1 
       (.I0(in_tdata[44]),
        .I1(state__0[0]),
        .I2(in_tdata[40]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[44]),
        .O(\out_tdata[44]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[44]_INST_0_i_2 
       (.I0(\out_tdata[44]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[44]_INST_0_i_5_n_0 ),
        .O(\out_tdata[44]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[44]_INST_0_i_3 
       (.I0(in_tdata[60]),
        .I1(in_tdata[40]),
        .I2(data_shift_next[1]),
        .I3(\packed_data_reg_n_0_[44] ),
        .I4(data_shift_next[2]),
        .I5(in_tdata[20]),
        .O(p_2_in__0[44]));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[44]_INST_0_i_4 
       (.I0(in_tdata[24]),
        .I1(data_shift_next[2]),
        .I2(\packed_data_reg_n_0_[44] ),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[44]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[44]_INST_0_i_5 
       (.I0(samp32[12]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[8]),
        .I4(\out_tdata[44]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[44]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[44]_INST_0_i_6 
       (.I0(in_tdata[24]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[12]),
        .O(samp32[12]));
  MUXF7 \out_tdata[44]_INST_0_i_7 
       (.I0(\out_tdata[44]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[44]_INST_0_i_9_n_0 ),
        .O(\out_tdata[44]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[44]_INST_0_i_8 
       (.I0(\packed_data_reg_n_0_[44] ),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[28]),
        .O(\out_tdata[44]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[44]_INST_0_i_9 
       (.I0(in_tdata[8]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[44] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[20]),
        .O(\out_tdata[44]_INST_0_i_9_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[45]_INST_0 
       (.I0(in_tdata[45]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[45]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[45]_INST_0_i_2_n_0 ),
        .O(out_tdata[45]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[45]_INST_0_i_1 
       (.I0(in_tdata[45]),
        .I1(state__0[0]),
        .I2(in_tdata[41]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[45]),
        .O(\out_tdata[45]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[45]_INST_0_i_2 
       (.I0(\out_tdata[45]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[45]_INST_0_i_5_n_0 ),
        .O(\out_tdata[45]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[45]_INST_0_i_3 
       (.I0(in_tdata[61]),
        .I1(in_tdata[41]),
        .I2(data_shift_next[1]),
        .I3(\packed_data_reg_n_0_[45] ),
        .I4(data_shift_next[2]),
        .I5(in_tdata[21]),
        .O(p_2_in__0[45]));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[45]_INST_0_i_4 
       (.I0(in_tdata[25]),
        .I1(data_shift_next[2]),
        .I2(\packed_data_reg_n_0_[45] ),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[45]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[45]_INST_0_i_5 
       (.I0(samp32[13]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[9]),
        .I4(\out_tdata[45]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[45]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[45]_INST_0_i_6 
       (.I0(in_tdata[25]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[13]),
        .O(samp32[13]));
  MUXF7 \out_tdata[45]_INST_0_i_7 
       (.I0(\out_tdata[45]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[45]_INST_0_i_9_n_0 ),
        .O(\out_tdata[45]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[45]_INST_0_i_8 
       (.I0(\packed_data_reg_n_0_[45] ),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[29]),
        .O(\out_tdata[45]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[45]_INST_0_i_9 
       (.I0(in_tdata[9]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[45] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[21]),
        .O(\out_tdata[45]_INST_0_i_9_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[46]_INST_0 
       (.I0(in_tdata[46]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[46]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[46]_INST_0_i_2_n_0 ),
        .O(out_tdata[46]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[46]_INST_0_i_1 
       (.I0(in_tdata[46]),
        .I1(state__0[0]),
        .I2(in_tdata[42]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[46]),
        .O(\out_tdata[46]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[46]_INST_0_i_2 
       (.I0(\out_tdata[46]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[46]_INST_0_i_5_n_0 ),
        .O(\out_tdata[46]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[46]_INST_0_i_3 
       (.I0(in_tdata[62]),
        .I1(in_tdata[42]),
        .I2(data_shift_next[1]),
        .I3(\packed_data_reg_n_0_[46] ),
        .I4(data_shift_next[2]),
        .I5(in_tdata[22]),
        .O(p_2_in__0[46]));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[46]_INST_0_i_4 
       (.I0(in_tdata[26]),
        .I1(data_shift_next[2]),
        .I2(\packed_data_reg_n_0_[46] ),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[46]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[46]_INST_0_i_5 
       (.I0(samp32[14]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[10]),
        .I4(\out_tdata[46]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[46]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[46]_INST_0_i_6 
       (.I0(in_tdata[26]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[14]),
        .O(samp32[14]));
  MUXF7 \out_tdata[46]_INST_0_i_7 
       (.I0(\out_tdata[46]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[46]_INST_0_i_9_n_0 ),
        .O(\out_tdata[46]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[46]_INST_0_i_8 
       (.I0(\packed_data_reg_n_0_[46] ),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[30]),
        .O(\out_tdata[46]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[46]_INST_0_i_9 
       (.I0(in_tdata[10]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[46] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[22]),
        .O(\out_tdata[46]_INST_0_i_9_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[47]_INST_0 
       (.I0(in_tdata[47]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[47]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[47]_INST_0_i_2_n_0 ),
        .O(out_tdata[47]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[47]_INST_0_i_1 
       (.I0(in_tdata[47]),
        .I1(state__0[0]),
        .I2(in_tdata[43]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[47]),
        .O(\out_tdata[47]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[47]_INST_0_i_2 
       (.I0(\out_tdata[47]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[47]_INST_0_i_5_n_0 ),
        .O(\out_tdata[47]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[47]_INST_0_i_3 
       (.I0(in_tdata[63]),
        .I1(in_tdata[43]),
        .I2(data_shift_next[1]),
        .I3(\packed_data_reg_n_0_[47] ),
        .I4(data_shift_next[2]),
        .I5(in_tdata[23]),
        .O(p_2_in__0[47]));
  LUT6 #(
    .INIT(64'h00000000F0F030E2)) 
    \out_tdata[47]_INST_0_i_4 
       (.I0(in_tdata[27]),
        .I1(data_shift_next[2]),
        .I2(\packed_data_reg_n_0_[47] ),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[47]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[47]_INST_0_i_5 
       (.I0(samp32[15]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[11]),
        .I4(\out_tdata[47]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[47]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[47]_INST_0_i_6 
       (.I0(in_tdata[27]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[15]),
        .O(samp32[15]));
  MUXF7 \out_tdata[47]_INST_0_i_7 
       (.I0(\out_tdata[47]_INST_0_i_8_n_0 ),
        .I1(\out_tdata[47]_INST_0_i_9_n_0 ),
        .O(\out_tdata[47]_INST_0_i_7_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[47]_INST_0_i_8 
       (.I0(\packed_data_reg_n_0_[47] ),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(in_tdata[31]),
        .O(\out_tdata[47]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[47]_INST_0_i_9 
       (.I0(in_tdata[11]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[47] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[23]),
        .O(\out_tdata[47]_INST_0_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[48]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[48]),
        .I2(state__0[0]),
        .I3(in14[48]),
        .I4(state__0[2]),
        .I5(\out_tdata[48]_INST_0_i_2_n_0 ),
        .O(out_tdata[48]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[48]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[44]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[4]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[24]),
        .O(in14[48]));
  MUXF7 \out_tdata[48]_INST_0_i_2 
       (.I0(\out_tdata[48]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[48]_INST_0_i_4_n_0 ),
        .O(\out_tdata[48]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[48]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[48] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[4]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[48]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[48]_INST_0_i_4 
       (.I0(samp32[16]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[12]),
        .I4(\out_tdata[48]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[48]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[48]_INST_0_i_5 
       (.I0(in_tdata[36]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[16]),
        .O(samp32[16]));
  MUXF7 \out_tdata[48]_INST_0_i_6 
       (.I0(\out_tdata[48]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[48]_INST_0_i_8_n_0 ),
        .O(\out_tdata[48]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[48]_INST_0_i_7 
       (.I0(in_tdata[4]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[48] ),
        .O(\out_tdata[48]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[48]_INST_0_i_8 
       (.I0(in_tdata[12]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[48] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[24]),
        .O(\out_tdata[48]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[49]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[49]),
        .I2(state__0[0]),
        .I3(in14[49]),
        .I4(state__0[2]),
        .I5(\out_tdata[49]_INST_0_i_2_n_0 ),
        .O(out_tdata[49]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[49]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[45]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[5]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[25]),
        .O(in14[49]));
  MUXF7 \out_tdata[49]_INST_0_i_2 
       (.I0(\out_tdata[49]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[49]_INST_0_i_4_n_0 ),
        .O(\out_tdata[49]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[49]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[49] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[5]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[49]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[49]_INST_0_i_4 
       (.I0(samp32[17]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[13]),
        .I4(\out_tdata[49]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[49]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[49]_INST_0_i_5 
       (.I0(in_tdata[37]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[17]),
        .O(samp32[17]));
  MUXF7 \out_tdata[49]_INST_0_i_6 
       (.I0(\out_tdata[49]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[49]_INST_0_i_8_n_0 ),
        .O(\out_tdata[49]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[49]_INST_0_i_7 
       (.I0(in_tdata[5]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[49] ),
        .O(\out_tdata[49]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[49]_INST_0_i_8 
       (.I0(in_tdata[13]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[49] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[25]),
        .O(\out_tdata[49]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[4]_INST_0 
       (.I0(in_tdata[4]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[4]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[4]_INST_0_i_2_n_0 ),
        .O(out_tdata[4]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[4]_INST_0_i_1 
       (.I0(in_tdata[4]),
        .I1(state__0[0]),
        .I2(p_1_in[4]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[4]),
        .O(\out_tdata[4]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[4]_INST_0_i_2 
       (.I0(\out_tdata[4]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[4]_INST_0_i_5_n_0 ),
        .O(\out_tdata[4]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[4]_INST_0_i_3 
       (.I0(in_tdata[8]),
        .I1(p_1_in[4]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[4]),
        .I4(data_shift_next[2]),
        .I5(data3[4]),
        .O(p_2_in__0[4]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[4]_INST_0_i_4 
       (.I0(in_tdata[8]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[4]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[4]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[4]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[4]),
        .I2(state__0[0]),
        .I3(p_1_in__0[4]),
        .I4(\out_tdata[4]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[4]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[4]_INST_0_i_6 
       (.I0(\out_tdata[4]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[4]_INST_0_i_8_n_0 ),
        .O(\out_tdata[4]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[4]_INST_0_i_7 
       (.I0(data3[4]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[4]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[8]),
        .O(\out_tdata[4]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[4]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(\samp24_last_reg_n_0_[12] ),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[4]),
        .O(\out_tdata[4]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[50]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[50]),
        .I2(state__0[0]),
        .I3(in14[50]),
        .I4(state__0[2]),
        .I5(\out_tdata[50]_INST_0_i_2_n_0 ),
        .O(out_tdata[50]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[50]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[46]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[6]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[26]),
        .O(in14[50]));
  MUXF7 \out_tdata[50]_INST_0_i_2 
       (.I0(\out_tdata[50]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[50]_INST_0_i_4_n_0 ),
        .O(\out_tdata[50]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[50]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[50] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[6]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[50]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[50]_INST_0_i_4 
       (.I0(samp32[18]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[14]),
        .I4(\out_tdata[50]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[50]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[50]_INST_0_i_5 
       (.I0(in_tdata[38]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[18]),
        .O(samp32[18]));
  MUXF7 \out_tdata[50]_INST_0_i_6 
       (.I0(\out_tdata[50]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[50]_INST_0_i_8_n_0 ),
        .O(\out_tdata[50]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[50]_INST_0_i_7 
       (.I0(in_tdata[6]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[50] ),
        .O(\out_tdata[50]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[50]_INST_0_i_8 
       (.I0(in_tdata[14]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[50] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[26]),
        .O(\out_tdata[50]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[51]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[51]),
        .I2(state__0[0]),
        .I3(in14[51]),
        .I4(state__0[2]),
        .I5(\out_tdata[51]_INST_0_i_2_n_0 ),
        .O(out_tdata[51]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[51]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[47]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[7]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[27]),
        .O(in14[51]));
  MUXF7 \out_tdata[51]_INST_0_i_2 
       (.I0(\out_tdata[51]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[51]_INST_0_i_4_n_0 ),
        .O(\out_tdata[51]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[51]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[51] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[7]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[51]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[51]_INST_0_i_4 
       (.I0(samp32[19]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[15]),
        .I4(\out_tdata[51]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[51]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[51]_INST_0_i_5 
       (.I0(in_tdata[39]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[19]),
        .O(samp32[19]));
  MUXF7 \out_tdata[51]_INST_0_i_6 
       (.I0(\out_tdata[51]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[51]_INST_0_i_8_n_0 ),
        .O(\out_tdata[51]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[51]_INST_0_i_7 
       (.I0(in_tdata[7]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[51] ),
        .O(\out_tdata[51]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[51]_INST_0_i_8 
       (.I0(in_tdata[15]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[51] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[27]),
        .O(\out_tdata[51]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[52]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[52]),
        .I2(state__0[0]),
        .I3(in14[52]),
        .I4(state__0[2]),
        .I5(\out_tdata[52]_INST_0_i_2_n_0 ),
        .O(out_tdata[52]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[52]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[52]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[8]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[28]),
        .O(in14[52]));
  MUXF7 \out_tdata[52]_INST_0_i_2 
       (.I0(\out_tdata[52]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[52]_INST_0_i_4_n_0 ),
        .O(\out_tdata[52]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[52]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[52] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[8]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[52]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[52]_INST_0_i_4 
       (.I0(samp32[20]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[20]),
        .I4(\out_tdata[52]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[52]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[52]_INST_0_i_5 
       (.I0(in_tdata[40]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[20]),
        .O(samp32[20]));
  MUXF7 \out_tdata[52]_INST_0_i_6 
       (.I0(\out_tdata[52]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[52]_INST_0_i_8_n_0 ),
        .O(\out_tdata[52]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[52]_INST_0_i_7 
       (.I0(in_tdata[8]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[52] ),
        .O(\out_tdata[52]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[52]_INST_0_i_8 
       (.I0(in_tdata[20]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[52] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[28]),
        .O(\out_tdata[52]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[53]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[53]),
        .I2(state__0[0]),
        .I3(in14[53]),
        .I4(state__0[2]),
        .I5(\out_tdata[53]_INST_0_i_2_n_0 ),
        .O(out_tdata[53]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[53]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[53]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[9]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[29]),
        .O(in14[53]));
  MUXF7 \out_tdata[53]_INST_0_i_2 
       (.I0(\out_tdata[53]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[53]_INST_0_i_4_n_0 ),
        .O(\out_tdata[53]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[53]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[53] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[9]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[53]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[53]_INST_0_i_4 
       (.I0(samp32[21]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[21]),
        .I4(\out_tdata[53]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[53]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[53]_INST_0_i_5 
       (.I0(in_tdata[41]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[21]),
        .O(samp32[21]));
  MUXF7 \out_tdata[53]_INST_0_i_6 
       (.I0(\out_tdata[53]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[53]_INST_0_i_8_n_0 ),
        .O(\out_tdata[53]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[53]_INST_0_i_7 
       (.I0(in_tdata[9]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[53] ),
        .O(\out_tdata[53]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[53]_INST_0_i_8 
       (.I0(in_tdata[21]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[53] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[29]),
        .O(\out_tdata[53]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[54]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[54]),
        .I2(state__0[0]),
        .I3(in14[54]),
        .I4(state__0[2]),
        .I5(\out_tdata[54]_INST_0_i_2_n_0 ),
        .O(out_tdata[54]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[54]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[54]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[10]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[30]),
        .O(in14[54]));
  MUXF7 \out_tdata[54]_INST_0_i_2 
       (.I0(\out_tdata[54]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[54]_INST_0_i_4_n_0 ),
        .O(\out_tdata[54]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[54]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[54] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[10]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[54]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[54]_INST_0_i_4 
       (.I0(samp32[22]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[22]),
        .I4(\out_tdata[54]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[54]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[54]_INST_0_i_5 
       (.I0(in_tdata[42]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[22]),
        .O(samp32[22]));
  MUXF7 \out_tdata[54]_INST_0_i_6 
       (.I0(\out_tdata[54]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[54]_INST_0_i_8_n_0 ),
        .O(\out_tdata[54]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[54]_INST_0_i_7 
       (.I0(in_tdata[10]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[54] ),
        .O(\out_tdata[54]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[54]_INST_0_i_8 
       (.I0(in_tdata[22]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[54] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[30]),
        .O(\out_tdata[54]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[55]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[55]),
        .I2(state__0[0]),
        .I3(in14[55]),
        .I4(state__0[2]),
        .I5(\out_tdata[55]_INST_0_i_2_n_0 ),
        .O(out_tdata[55]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[55]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[55]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[11]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[31]),
        .O(in14[55]));
  MUXF7 \out_tdata[55]_INST_0_i_2 
       (.I0(\out_tdata[55]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[55]_INST_0_i_4_n_0 ),
        .O(\out_tdata[55]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[55]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[55] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[11]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[55]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[55]_INST_0_i_4 
       (.I0(samp32[23]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[23]),
        .I4(\out_tdata[55]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[55]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[55]_INST_0_i_5 
       (.I0(in_tdata[43]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[23]),
        .O(samp32[23]));
  MUXF7 \out_tdata[55]_INST_0_i_6 
       (.I0(\out_tdata[55]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[55]_INST_0_i_8_n_0 ),
        .O(\out_tdata[55]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[55]_INST_0_i_7 
       (.I0(in_tdata[11]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[55] ),
        .O(\out_tdata[55]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[55]_INST_0_i_8 
       (.I0(in_tdata[23]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\packed_data_reg_n_0_[55] ),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[31]),
        .O(\out_tdata[55]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[56]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[56]),
        .I2(state__0[0]),
        .I3(in14[56]),
        .I4(state__0[2]),
        .I5(\out_tdata[56]_INST_0_i_2_n_0 ),
        .O(out_tdata[56]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[56]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[56]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[12]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[36]),
        .O(in14[56]));
  MUXF7 \out_tdata[56]_INST_0_i_2 
       (.I0(\out_tdata[56]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[56]_INST_0_i_4_n_0 ),
        .O(\out_tdata[56]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[56]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[56] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[20]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[56]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[56]_INST_0_i_4 
       (.I0(samp32[24]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[24]),
        .I4(\out_tdata[56]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[56]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[56]_INST_0_i_5 
       (.I0(in_tdata[52]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[24]),
        .O(samp32[24]));
  MUXF7 \out_tdata[56]_INST_0_i_6 
       (.I0(\out_tdata[56]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[56]_INST_0_i_8_n_0 ),
        .O(\out_tdata[56]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[56]_INST_0_i_7 
       (.I0(in_tdata[12]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[56] ),
        .O(\out_tdata[56]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[56]_INST_0_i_8 
       (.I0(in_tdata[24]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(in_tdata[4]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(\packed_data_reg_n_0_[56] ),
        .O(\out_tdata[56]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[57]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[57]),
        .I2(state__0[0]),
        .I3(in14[57]),
        .I4(state__0[2]),
        .I5(\out_tdata[57]_INST_0_i_2_n_0 ),
        .O(out_tdata[57]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[57]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[57]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[13]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[37]),
        .O(in14[57]));
  MUXF7 \out_tdata[57]_INST_0_i_2 
       (.I0(\out_tdata[57]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[57]_INST_0_i_4_n_0 ),
        .O(\out_tdata[57]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[57]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[57] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[21]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[57]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[57]_INST_0_i_4 
       (.I0(samp32[25]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[25]),
        .I4(\out_tdata[57]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[57]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[57]_INST_0_i_5 
       (.I0(in_tdata[53]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[25]),
        .O(samp32[25]));
  MUXF7 \out_tdata[57]_INST_0_i_6 
       (.I0(\out_tdata[57]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[57]_INST_0_i_8_n_0 ),
        .O(\out_tdata[57]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[57]_INST_0_i_7 
       (.I0(in_tdata[13]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[57] ),
        .O(\out_tdata[57]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[57]_INST_0_i_8 
       (.I0(in_tdata[25]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(in_tdata[5]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(\packed_data_reg_n_0_[57] ),
        .O(\out_tdata[57]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[58]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[58]),
        .I2(state__0[0]),
        .I3(in14[58]),
        .I4(state__0[2]),
        .I5(\out_tdata[58]_INST_0_i_2_n_0 ),
        .O(out_tdata[58]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[58]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[58]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[14]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[38]),
        .O(in14[58]));
  MUXF7 \out_tdata[58]_INST_0_i_2 
       (.I0(\out_tdata[58]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[58]_INST_0_i_4_n_0 ),
        .O(\out_tdata[58]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[58]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[58] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[22]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[58]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[58]_INST_0_i_4 
       (.I0(samp32[26]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[26]),
        .I4(\out_tdata[58]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[58]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[58]_INST_0_i_5 
       (.I0(in_tdata[54]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[26]),
        .O(samp32[26]));
  MUXF7 \out_tdata[58]_INST_0_i_6 
       (.I0(\out_tdata[58]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[58]_INST_0_i_8_n_0 ),
        .O(\out_tdata[58]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[58]_INST_0_i_7 
       (.I0(in_tdata[14]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[58] ),
        .O(\out_tdata[58]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[58]_INST_0_i_8 
       (.I0(in_tdata[26]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(in_tdata[6]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(\packed_data_reg_n_0_[58] ),
        .O(\out_tdata[58]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[59]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[59]),
        .I2(state__0[0]),
        .I3(in14[59]),
        .I4(state__0[2]),
        .I5(\out_tdata[59]_INST_0_i_2_n_0 ),
        .O(out_tdata[59]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[59]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[59]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[15]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[39]),
        .O(in14[59]));
  MUXF7 \out_tdata[59]_INST_0_i_2 
       (.I0(\out_tdata[59]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[59]_INST_0_i_4_n_0 ),
        .O(\out_tdata[59]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[59]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[59] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[23]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[59]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[59]_INST_0_i_4 
       (.I0(samp32[27]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[27]),
        .I4(\out_tdata[59]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[59]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[59]_INST_0_i_5 
       (.I0(in_tdata[55]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[27]),
        .O(samp32[27]));
  MUXF7 \out_tdata[59]_INST_0_i_6 
       (.I0(\out_tdata[59]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[59]_INST_0_i_8_n_0 ),
        .O(\out_tdata[59]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[59]_INST_0_i_7 
       (.I0(in_tdata[15]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[59] ),
        .O(\out_tdata[59]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[59]_INST_0_i_8 
       (.I0(in_tdata[27]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(in_tdata[7]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(\packed_data_reg_n_0_[59] ),
        .O(\out_tdata[59]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[5]_INST_0 
       (.I0(in_tdata[5]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[5]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[5]_INST_0_i_2_n_0 ),
        .O(out_tdata[5]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[5]_INST_0_i_1 
       (.I0(in_tdata[5]),
        .I1(state__0[0]),
        .I2(p_1_in[5]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[5]),
        .O(\out_tdata[5]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[5]_INST_0_i_2 
       (.I0(\out_tdata[5]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[5]_INST_0_i_5_n_0 ),
        .O(\out_tdata[5]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[5]_INST_0_i_3 
       (.I0(in_tdata[9]),
        .I1(p_1_in[5]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[5]),
        .I4(data_shift_next[2]),
        .I5(data3[5]),
        .O(p_2_in__0[5]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[5]_INST_0_i_4 
       (.I0(in_tdata[9]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[5]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[5]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[5]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[5]),
        .I2(state__0[0]),
        .I3(p_1_in__0[5]),
        .I4(\out_tdata[5]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[5]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[5]_INST_0_i_6 
       (.I0(\out_tdata[5]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[5]_INST_0_i_8_n_0 ),
        .O(\out_tdata[5]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[5]_INST_0_i_7 
       (.I0(data3[5]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[5]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[9]),
        .O(\out_tdata[5]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[5]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(\samp24_last_reg_n_0_[13] ),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[5]),
        .O(\out_tdata[5]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[60]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[60]),
        .I2(state__0[0]),
        .I3(in14[60]),
        .I4(state__0[2]),
        .I5(\out_tdata[60]_INST_0_i_2_n_0 ),
        .O(out_tdata[60]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[60]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[60]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[20]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[40]),
        .O(in14[60]));
  MUXF7 \out_tdata[60]_INST_0_i_2 
       (.I0(\out_tdata[60]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[60]_INST_0_i_4_n_0 ),
        .O(\out_tdata[60]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[60]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[60] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[24]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[60]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[60]_INST_0_i_4 
       (.I0(samp32[28]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[28]),
        .I4(\out_tdata[60]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[60]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[60]_INST_0_i_5 
       (.I0(in_tdata[56]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[28]),
        .O(samp32[28]));
  MUXF7 \out_tdata[60]_INST_0_i_6 
       (.I0(\out_tdata[60]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[60]_INST_0_i_8_n_0 ),
        .O(\out_tdata[60]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[60]_INST_0_i_7 
       (.I0(in_tdata[20]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[60] ),
        .O(\out_tdata[60]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[60]_INST_0_i_8 
       (.I0(in_tdata[28]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(in_tdata[8]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(\packed_data_reg_n_0_[60] ),
        .O(\out_tdata[60]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[61]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[61]),
        .I2(state__0[0]),
        .I3(in14[61]),
        .I4(state__0[2]),
        .I5(\out_tdata[61]_INST_0_i_2_n_0 ),
        .O(out_tdata[61]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[61]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[61]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[21]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[41]),
        .O(in14[61]));
  MUXF7 \out_tdata[61]_INST_0_i_2 
       (.I0(\out_tdata[61]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[61]_INST_0_i_4_n_0 ),
        .O(\out_tdata[61]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[61]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[61] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[25]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[61]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[61]_INST_0_i_4 
       (.I0(samp32[29]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[29]),
        .I4(\out_tdata[61]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[61]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[61]_INST_0_i_5 
       (.I0(in_tdata[57]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[29]),
        .O(samp32[29]));
  MUXF7 \out_tdata[61]_INST_0_i_6 
       (.I0(\out_tdata[61]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[61]_INST_0_i_8_n_0 ),
        .O(\out_tdata[61]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[61]_INST_0_i_7 
       (.I0(in_tdata[21]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[61] ),
        .O(\out_tdata[61]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[61]_INST_0_i_8 
       (.I0(in_tdata[29]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(in_tdata[9]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(\packed_data_reg_n_0_[61] ),
        .O(\out_tdata[61]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[62]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[62]),
        .I2(state__0[0]),
        .I3(in14[62]),
        .I4(state__0[2]),
        .I5(\out_tdata[62]_INST_0_i_2_n_0 ),
        .O(out_tdata[62]));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[62]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[62]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[22]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[42]),
        .O(in14[62]));
  MUXF7 \out_tdata[62]_INST_0_i_2 
       (.I0(\out_tdata[62]_INST_0_i_3_n_0 ),
        .I1(\out_tdata[62]_INST_0_i_4_n_0 ),
        .O(\out_tdata[62]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[62]_INST_0_i_3 
       (.I0(\packed_data_reg_n_0_[62] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[26]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[62]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[62]_INST_0_i_4 
       (.I0(samp32[30]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[30]),
        .I4(\out_tdata[62]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[62]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[62]_INST_0_i_5 
       (.I0(in_tdata[58]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[30]),
        .O(samp32[30]));
  MUXF7 \out_tdata[62]_INST_0_i_6 
       (.I0(\out_tdata[62]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[62]_INST_0_i_8_n_0 ),
        .O(\out_tdata[62]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[62]_INST_0_i_7 
       (.I0(in_tdata[22]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[62] ),
        .O(\out_tdata[62]_INST_0_i_7_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[62]_INST_0_i_8 
       (.I0(in_tdata[30]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(in_tdata[10]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(\packed_data_reg_n_0_[62] ),
        .O(\out_tdata[62]_INST_0_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hCDC8DDDDCDC88888)) 
    \out_tdata[63]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(in_tdata[63]),
        .I2(state__0[0]),
        .I3(in14[63]),
        .I4(state__0[2]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[63]));
  LUT3 #(
    .INIT(8'h02)) 
    \out_tdata[63]_INST_0_i_1 
       (.I0(state__0[0]),
        .I1(state__0[2]),
        .I2(state__0[1]),
        .O(\out_tdata[63]_INST_0_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h7)) 
    \out_tdata[63]_INST_0_i_10 
       (.I0(\data_shift_reg_n_0_[7] ),
        .I1(data_shift_next[7]),
        .O(\out_tdata[63]_INST_0_i_10_n_0 ));
  LUT6 #(
    .INIT(64'h7F7FFF7FFFFFFF7F)) 
    \out_tdata[63]_INST_0_i_11 
       (.I0(data_shift_next[6]),
        .I1(data_shift_next[4]),
        .I2(data_shift_next[5]),
        .I3(data_shift_next[1]),
        .I4(data_shift_next[2]),
        .I5(data_shift_next[3]),
        .O(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT4 #(
    .INIT(16'h3808)) 
    \out_tdata[63]_INST_0_i_12 
       (.I0(in_tdata[23]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I3(\packed_data_reg_n_0_[63] ),
        .O(\out_tdata[63]_INST_0_i_12_n_0 ));
  LUT5 #(
    .INIT(32'hB833B800)) 
    \out_tdata[63]_INST_0_i_13 
       (.I0(in_tdata[31]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(in_tdata[11]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(\packed_data_reg_n_0_[63] ),
        .O(\out_tdata[63]_INST_0_i_13_n_0 ));
  LUT6 #(
    .INIT(64'h3FFFFFEFFFFFFFEF)) 
    \out_tdata[63]_INST_0_i_14 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[4]),
        .I2(data_shift_next[6]),
        .I3(data_shift_next[2]),
        .I4(data_shift_next[3]),
        .I5(data_shift_next[5]),
        .O(\out_tdata[63]_INST_0_i_14_n_0 ));
  LUT6 #(
    .INIT(64'hF6F6F6F666F6F6F6)) 
    \out_tdata[63]_INST_0_i_15 
       (.I0(data_shift_next[2]),
        .I1(data_shift_next[1]),
        .I2(\out_tdata[63]_INST_0_i_16_n_0 ),
        .I3(data_shift_next[6]),
        .I4(data_shift_next[5]),
        .I5(\out_tdata[63]_INST_0_i_4_n_0 ),
        .O(\out_tdata[63]_INST_0_i_15_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFEFFE)) 
    \out_tdata[63]_INST_0_i_16 
       (.I0(data_shift_next[2]),
        .I1(data_shift_next[4]),
        .I2(data_shift_next[5]),
        .I3(data_shift_next[6]),
        .I4(data_shift_next[3]),
        .O(\out_tdata[63]_INST_0_i_16_n_0 ));
  LUT6 #(
    .INIT(64'h8D88CDCD8D88C8C8)) 
    \out_tdata[63]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[63]),
        .I2(data_shift_next[1]),
        .I3(in_tdata[23]),
        .I4(data_shift_next[2]),
        .I5(in_tdata[43]),
        .O(in14[63]));
  MUXF7 \out_tdata[63]_INST_0_i_3 
       (.I0(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I1(\out_tdata[63]_INST_0_i_6_n_0 ),
        .O(\out_tdata[63]_INST_0_i_3_n_0 ),
        .S(state__0[1]));
  LUT2 #(
    .INIT(4'h7)) 
    \out_tdata[63]_INST_0_i_4 
       (.I0(data_shift_next[4]),
        .I1(data_shift_next[3]),
        .O(\out_tdata[63]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FF002E22)) 
    \out_tdata[63]_INST_0_i_5 
       (.I0(\packed_data_reg_n_0_[63] ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(in_tdata[27]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[63]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h8F808F808F8F8080)) 
    \out_tdata[63]_INST_0_i_6 
       (.I0(samp32[31]),
        .I1(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I2(state__0[0]),
        .I3(in_tdata[31]),
        .I4(\out_tdata[63]_INST_0_i_9_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[63]_INST_0_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \out_tdata[63]_INST_0_i_7 
       (.I0(in_tdata[59]),
        .I1(in_tctrl[3]),
        .I2(in_tdata[31]),
        .O(samp32[31]));
  LUT2 #(
    .INIT(4'h7)) 
    \out_tdata[63]_INST_0_i_8 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[2]),
        .O(\out_tdata[63]_INST_0_i_8_n_0 ));
  MUXF7 \out_tdata[63]_INST_0_i_9 
       (.I0(\out_tdata[63]_INST_0_i_12_n_0 ),
        .I1(\out_tdata[63]_INST_0_i_13_n_0 ),
        .O(\out_tdata[63]_INST_0_i_9_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[6]_INST_0 
       (.I0(in_tdata[6]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[6]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[6]_INST_0_i_2_n_0 ),
        .O(out_tdata[6]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[6]_INST_0_i_1 
       (.I0(in_tdata[6]),
        .I1(state__0[0]),
        .I2(p_1_in[6]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[6]),
        .O(\out_tdata[6]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[6]_INST_0_i_2 
       (.I0(\out_tdata[6]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[6]_INST_0_i_5_n_0 ),
        .O(\out_tdata[6]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[6]_INST_0_i_3 
       (.I0(in_tdata[10]),
        .I1(p_1_in[6]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[6]),
        .I4(data_shift_next[2]),
        .I5(data3[6]),
        .O(p_2_in__0[6]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[6]_INST_0_i_4 
       (.I0(in_tdata[10]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[6]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[6]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[6]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[6]),
        .I2(state__0[0]),
        .I3(p_1_in__0[6]),
        .I4(\out_tdata[6]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[6]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[6]_INST_0_i_6 
       (.I0(\out_tdata[6]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[6]_INST_0_i_8_n_0 ),
        .O(\out_tdata[6]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[6]_INST_0_i_7 
       (.I0(data3[6]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[6]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[10]),
        .O(\out_tdata[6]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[6]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(\samp24_last_reg_n_0_[14] ),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[6]),
        .O(\out_tdata[6]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[7]_INST_0 
       (.I0(in_tdata[7]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[7]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[7]_INST_0_i_2_n_0 ),
        .O(out_tdata[7]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[7]_INST_0_i_1 
       (.I0(in_tdata[7]),
        .I1(state__0[0]),
        .I2(p_1_in[7]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[7]),
        .O(\out_tdata[7]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[7]_INST_0_i_2 
       (.I0(\out_tdata[7]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[7]_INST_0_i_5_n_0 ),
        .O(\out_tdata[7]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[7]_INST_0_i_3 
       (.I0(in_tdata[11]),
        .I1(p_1_in[7]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[7]),
        .I4(data_shift_next[2]),
        .I5(data3[7]),
        .O(p_2_in__0[7]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[7]_INST_0_i_4 
       (.I0(in_tdata[11]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[7]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[7]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[7]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[7]),
        .I2(state__0[0]),
        .I3(p_1_in__0[7]),
        .I4(\out_tdata[7]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[7]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[7]_INST_0_i_6 
       (.I0(\out_tdata[7]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[7]_INST_0_i_8_n_0 ),
        .O(\out_tdata[7]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[7]_INST_0_i_7 
       (.I0(data3[7]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[7]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[11]),
        .O(\out_tdata[7]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[7]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(\samp24_last_reg_n_0_[15] ),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[7]),
        .O(\out_tdata[7]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[8]_INST_0 
       (.I0(in_tdata[8]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[8]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[8]_INST_0_i_2_n_0 ),
        .O(out_tdata[8]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[8]_INST_0_i_1 
       (.I0(in_tdata[8]),
        .I1(state__0[0]),
        .I2(p_1_in[8]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[8]),
        .O(\out_tdata[8]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[8]_INST_0_i_2 
       (.I0(\out_tdata[8]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[8]_INST_0_i_5_n_0 ),
        .O(\out_tdata[8]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[8]_INST_0_i_3 
       (.I0(in_tdata[12]),
        .I1(p_1_in[8]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[8]),
        .I4(data_shift_next[2]),
        .I5(\samp48_last_reg_n_0_[24] ),
        .O(p_2_in__0[8]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[8]_INST_0_i_4 
       (.I0(in_tdata[20]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[8]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[8]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[8]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[8]),
        .I2(state__0[0]),
        .I3(p_1_in__0[8]),
        .I4(\out_tdata[8]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[8]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[8]_INST_0_i_6 
       (.I0(\out_tdata[8]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[8]_INST_0_i_8_n_0 ),
        .O(\out_tdata[8]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[8]_INST_0_i_7 
       (.I0(in_tdata[4]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[8]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[12]),
        .O(\out_tdata[8]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[8]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(data3[0]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[8]),
        .O(\out_tdata[8]_INST_0_i_8_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[9]_INST_0 
       (.I0(in_tdata[9]),
        .I1(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[9]_INST_0_i_1_n_0 ),
        .I3(state__0[2]),
        .I4(\out_tdata[9]_INST_0_i_2_n_0 ),
        .O(out_tdata[9]));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \out_tdata[9]_INST_0_i_1 
       (.I0(in_tdata[9]),
        .I1(state__0[0]),
        .I2(p_1_in[9]),
        .I3(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I4(p_2_in__0[9]),
        .O(\out_tdata[9]_INST_0_i_1_n_0 ));
  MUXF7 \out_tdata[9]_INST_0_i_2 
       (.I0(\out_tdata[9]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[9]_INST_0_i_5_n_0 ),
        .O(\out_tdata[9]_INST_0_i_2_n_0 ),
        .S(state__0[1]));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \out_tdata[9]_INST_0_i_3 
       (.I0(in_tdata[13]),
        .I1(p_1_in[9]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[9]),
        .I4(data_shift_next[2]),
        .I5(\samp48_last_reg_n_0_[25] ),
        .O(p_2_in__0[9]));
  LUT6 #(
    .INIT(64'h00000000FF00BF80)) 
    \out_tdata[9]_INST_0_i_4 
       (.I0(in_tdata[21]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(p_1_in__0[9]),
        .I4(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I5(state__0[0]),
        .O(\out_tdata[9]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEF40EF40EF4FE040)) 
    \out_tdata[9]_INST_0_i_5 
       (.I0(\out_tdata[63]_INST_0_i_8_n_0 ),
        .I1(samp32[9]),
        .I2(state__0[0]),
        .I3(p_1_in__0[9]),
        .I4(\out_tdata[9]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_10_n_0 ),
        .O(\out_tdata[9]_INST_0_i_5_n_0 ));
  MUXF7 \out_tdata[9]_INST_0_i_6 
       (.I0(\out_tdata[9]_INST_0_i_7_n_0 ),
        .I1(\out_tdata[9]_INST_0_i_8_n_0 ),
        .O(\out_tdata[9]_INST_0_i_6_n_0 ),
        .S(\out_tdata[63]_INST_0_i_11_n_0 ));
  LUT5 #(
    .INIT(32'hB8F3B8C0)) 
    \out_tdata[9]_INST_0_i_7 
       (.I0(in_tdata[5]),
        .I1(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I2(p_1_in__0[9]),
        .I3(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I4(in_tdata[13]),
        .O(\out_tdata[9]_INST_0_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hEF40)) 
    \out_tdata[9]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_15_n_0 ),
        .I1(data3[1]),
        .I2(\out_tdata[63]_INST_0_i_14_n_0 ),
        .I3(p_1_in__0[9]),
        .O(\out_tdata[9]_INST_0_i_8_n_0 ));
  LUT4 #(
    .INIT(16'hFFF7)) 
    \out_tkeep[1]_INST_0 
       (.I0(out_tlast),
        .I1(\out_tkeep[7]_INST_0_i_1_n_0 ),
        .I2(\out_tkeep[7]_INST_0_i_3_n_0 ),
        .I3(\out_tkeep[7]_INST_0_i_2_n_0 ),
        .O(\^out_tkeep [1]));
  LUT4 #(
    .INIT(16'hFF9F)) 
    \out_tkeep[2]_INST_0 
       (.I0(\out_tkeep[7]_INST_0_i_1_n_0 ),
        .I1(\out_tkeep[7]_INST_0_i_3_n_0 ),
        .I2(out_tlast),
        .I3(\out_tkeep[7]_INST_0_i_2_n_0 ),
        .O(\^out_tkeep [2]));
  LUT4 #(
    .INIT(16'hABFF)) 
    \out_tkeep[3]_INST_0 
       (.I0(\out_tkeep[7]_INST_0_i_2_n_0 ),
        .I1(\out_tkeep[7]_INST_0_i_3_n_0 ),
        .I2(\out_tkeep[7]_INST_0_i_1_n_0 ),
        .I3(out_tlast),
        .O(\^out_tkeep [3]));
  LUT4 #(
    .INIT(16'hDDD7)) 
    \out_tkeep[4]_INST_0 
       (.I0(out_tlast),
        .I1(\out_tkeep[7]_INST_0_i_2_n_0 ),
        .I2(\out_tkeep[7]_INST_0_i_3_n_0 ),
        .I3(\out_tkeep[7]_INST_0_i_1_n_0 ),
        .O(\^out_tkeep [4]));
  LUT4 #(
    .INIT(16'hC1FF)) 
    \out_tkeep[5]_INST_0 
       (.I0(\out_tkeep[7]_INST_0_i_1_n_0 ),
        .I1(\out_tkeep[7]_INST_0_i_2_n_0 ),
        .I2(\out_tkeep[7]_INST_0_i_3_n_0 ),
        .I3(out_tlast),
        .O(\^out_tkeep [5]));
  LUT4 #(
    .INIT(16'hD557)) 
    \out_tkeep[6]_INST_0 
       (.I0(out_tlast),
        .I1(\out_tkeep[7]_INST_0_i_3_n_0 ),
        .I2(\out_tkeep[7]_INST_0_i_2_n_0 ),
        .I3(\out_tkeep[7]_INST_0_i_1_n_0 ),
        .O(\^out_tkeep [6]));
  LUT4 #(
    .INIT(16'h5557)) 
    \out_tkeep[7]_INST_0 
       (.I0(out_tlast),
        .I1(\out_tkeep[7]_INST_0_i_1_n_0 ),
        .I2(\out_tkeep[7]_INST_0_i_2_n_0 ),
        .I3(\out_tkeep[7]_INST_0_i_3_n_0 ),
        .O(\^out_tkeep [7]));
  LUT6 #(
    .INIT(64'h00000000FFDF0020)) 
    \out_tkeep[7]_INST_0_i_1 
       (.I0(p_2_in15_out),
        .I1(state__0[0]),
        .I2(state__0[1]),
        .I3(state__0[2]),
        .I4(bytes_occ[0]),
        .I5(\out_tdata[63]_INST_0_i_1_n_0 ),
        .O(\out_tkeep[7]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000099969666)) 
    \out_tkeep[7]_INST_0_i_2 
       (.I0(bytes_occ[2]),
        .I1(\out_tkeep[7]_INST_0_i_4_n_0 ),
        .I2(\out_tkeep[7]_INST_0_i_5_n_0 ),
        .I3(bytes_occ[1]),
        .I4(\out_tkeep[7]_INST_0_i_6_n_0 ),
        .I5(\out_tdata[63]_INST_0_i_1_n_0 ),
        .O(\out_tkeep[7]_INST_0_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h0096)) 
    \out_tkeep[7]_INST_0_i_3 
       (.I0(bytes_occ[1]),
        .I1(\out_tkeep[7]_INST_0_i_5_n_0 ),
        .I2(\out_tkeep[7]_INST_0_i_6_n_0 ),
        .I3(\out_tdata[63]_INST_0_i_1_n_0 ),
        .O(\out_tkeep[7]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h0000200000200000)) 
    \out_tkeep[7]_INST_0_i_4 
       (.I0(out_tready),
        .I1(extra_cycle_reg_n_0),
        .I2(in_tvalid),
        .I3(state__0[1]),
        .I4(state__0[2]),
        .I5(state__0[0]),
        .O(\out_tkeep[7]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h0000002000200020)) 
    \out_tkeep[7]_INST_0_i_5 
       (.I0(out_tready),
        .I1(extra_cycle_reg_n_0),
        .I2(in_tvalid),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\out_tkeep[7]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h04000000)) 
    \out_tkeep[7]_INST_0_i_6 
       (.I0(state__0[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(p_2_in15_out),
        .I4(bytes_occ[0]),
        .O(\out_tkeep[7]_INST_0_i_6_n_0 ));
  LUT5 #(
    .INIT(32'hFFFF0040)) 
    out_tlast_INST_0
       (.I0(out_tlast_INST_0_i_1_n_0),
        .I1(in_tvalid),
        .I2(in_tlast),
        .I3(out_tlast_INST_0_i_2_n_0),
        .I4(extra_cycle_reg_n_0),
        .O(out_tlast));
  LUT6 #(
    .INIT(64'h8000000000000080)) 
    out_tlast_INST_0_i_1
       (.I0(data_shift_next[6]),
        .I1(data_shift_next[7]),
        .I2(out_tlast_INST_0_i_3_n_0),
        .I3(data_shift_next[4]),
        .I4(data_shift_next[3]),
        .I5(data_shift_next[5]),
        .O(out_tlast_INST_0_i_1_n_0));
  LUT5 #(
    .INIT(32'h00004000)) 
    out_tlast_INST_0_i_2
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[4]),
        .I2(data_shift_next[3]),
        .I3(out_tlast_INST_0_i_4_n_0),
        .I4(state__0[0]),
        .O(out_tlast_INST_0_i_2_n_0));
  LUT6 #(
    .INIT(64'h0000000000100000)) 
    out_tlast_INST_0_i_3
       (.I0(data_shift_next[1]),
        .I1(state__0[0]),
        .I2(state__0[1]),
        .I3(state__0[2]),
        .I4(\data_shift_reg_n_0_[7] ),
        .I5(data_shift_next[2]),
        .O(out_tlast_INST_0_i_3_n_0));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT2 #(
    .INIT(4'h2)) 
    out_tlast_INST_0_i_4
       (.I0(state__0[2]),
        .I1(state__0[1]),
        .O(out_tlast_INST_0_i_4_n_0));
  LUT3 #(
    .INIT(8'hAC)) 
    out_tvalid_INST_0
       (.I0(in_tvalid),
        .I1(out_tlast),
        .I2(out_tvalid_i1),
        .O(out_tvalid));
  LUT6 #(
    .INIT(64'h00000000FFFFFFFE)) 
    out_tvalid_INST_0_i_1
       (.I0(out_tvalid_INST_0_i_2_n_0),
        .I1(xfer_cycle4),
        .I2(out_tvalid_INST_0_i_4_n_0),
        .I3(out_tvalid_INST_0_i_5_n_0),
        .I4(out_tvalid_INST_0_i_6_n_0),
        .I5(extra_cycle_reg_n_0),
        .O(out_tvalid_i1));
  LUT6 #(
    .INIT(64'h8000000000000000)) 
    out_tvalid_INST_0_i_2
       (.I0(out_tlast_INST_0_i_3_n_0),
        .I1(data_shift_next[5]),
        .I2(data_shift_next[4]),
        .I3(data_shift_next[3]),
        .I4(data_shift_next[6]),
        .I5(data_shift_next[7]),
        .O(out_tvalid_INST_0_i_2_n_0));
  LUT5 #(
    .INIT(32'h00200000)) 
    out_tvalid_INST_0_i_3
       (.I0(state__0[1]),
        .I1(state__0[2]),
        .I2(state__0[0]),
        .I3(data_shift_next[1]),
        .I4(data_shift_next[2]),
        .O(xfer_cycle4));
  LUT6 #(
    .INIT(64'h0000000000000090)) 
    out_tvalid_INST_0_i_4
       (.I0(data_shift_next[6]),
        .I1(data_shift_next[7]),
        .I2(out_tlast_INST_0_i_3_n_0),
        .I3(data_shift_next[4]),
        .I4(data_shift_next[3]),
        .I5(data_shift_next[5]),
        .O(out_tvalid_INST_0_i_4_n_0));
  LUT6 #(
    .INIT(64'h8888CC8888888C88)) 
    out_tvalid_INST_0_i_5
       (.I0(state__0[0]),
        .I1(out_tlast_INST_0_i_4_n_0),
        .I2(data_shift_next[2]),
        .I3(data_shift_next[4]),
        .I4(data_shift_next[1]),
        .I5(data_shift_next[3]),
        .O(out_tvalid_INST_0_i_5_n_0));
  LUT5 #(
    .INIT(32'hAAAAAAAE)) 
    out_tvalid_INST_0_i_6
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(xfer_cycle6),
        .I2(state__0[2]),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .O(out_tvalid_INST_0_i_6_n_0));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT4 #(
    .INIT(16'h0010)) 
    out_tvalid_INST_0_i_7
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[3]),
        .I2(data_shift_next[4]),
        .I3(data_shift_next[2]),
        .O(xfer_cycle6));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[0]_i_1 
       (.I0(\out_tdata[0]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[0]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[0]_INST_0_i_4_n_0 ),
        .O(\packed_data[0]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[10]_i_1 
       (.I0(\out_tdata[10]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[10]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[10]_INST_0_i_4_n_0 ),
        .O(\packed_data[10]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[11]_i_1 
       (.I0(\out_tdata[11]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[11]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[11]_INST_0_i_4_n_0 ),
        .O(\packed_data[11]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[12]_i_1 
       (.I0(\out_tdata[12]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[12]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[12]_INST_0_i_4_n_0 ),
        .O(\packed_data[12]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[13]_i_1 
       (.I0(\out_tdata[13]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[13]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[13]_INST_0_i_4_n_0 ),
        .O(\packed_data[13]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[14]_i_1 
       (.I0(\out_tdata[14]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[14]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[14]_INST_0_i_4_n_0 ),
        .O(\packed_data[14]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[15]_i_1 
       (.I0(\out_tdata[15]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[15]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[15]_INST_0_i_4_n_0 ),
        .O(\packed_data[15]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[16]_i_1 
       (.I0(\out_tdata[16]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[16]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[16]_INST_0_i_4_n_0 ),
        .O(\packed_data[16]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[17]_i_1 
       (.I0(\out_tdata[17]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[17]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[17]_INST_0_i_4_n_0 ),
        .O(\packed_data[17]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[18]_i_1 
       (.I0(\out_tdata[18]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[18]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[18]_INST_0_i_4_n_0 ),
        .O(\packed_data[18]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[19]_i_1 
       (.I0(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[19]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[19]_INST_0_i_4_n_0 ),
        .O(\packed_data[19]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[1]_i_1 
       (.I0(\out_tdata[1]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[1]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[1]_INST_0_i_4_n_0 ),
        .O(\packed_data[1]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[20]_i_1 
       (.I0(\out_tdata[20]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[20]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[20]_INST_0_i_4_n_0 ),
        .O(\packed_data[20]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[21]_i_1 
       (.I0(\out_tdata[21]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[21]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[21]_INST_0_i_4_n_0 ),
        .O(\packed_data[21]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[22]_i_1 
       (.I0(\out_tdata[22]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[22]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[22]_INST_0_i_4_n_0 ),
        .O(\packed_data[22]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[23]_i_1 
       (.I0(\out_tdata[23]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[23]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[23]_INST_0_i_4_n_0 ),
        .O(\packed_data[23]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[24]_i_1 
       (.I0(\out_tdata[24]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[24]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[24]_INST_0_i_4_n_0 ),
        .O(\packed_data[24]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[25]_i_1 
       (.I0(\out_tdata[25]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[25]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[25]_INST_0_i_4_n_0 ),
        .O(\packed_data[25]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[26]_i_1 
       (.I0(\out_tdata[26]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[26]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[26]_INST_0_i_4_n_0 ),
        .O(\packed_data[26]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[27]_i_1 
       (.I0(\out_tdata[27]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[27]_INST_0_i_4_n_0 ),
        .O(\packed_data[27]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[28]_i_1 
       (.I0(\out_tdata[28]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[28]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[28]_INST_0_i_4_n_0 ),
        .O(\packed_data[28]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[29]_i_1 
       (.I0(\out_tdata[29]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[29]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[29]_INST_0_i_4_n_0 ),
        .O(\packed_data[29]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[2]_i_1 
       (.I0(\out_tdata[2]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[2]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[2]_INST_0_i_4_n_0 ),
        .O(\packed_data[2]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[30]_i_1 
       (.I0(\out_tdata[30]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[30]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[30]_INST_0_i_4_n_0 ),
        .O(\packed_data[30]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[31]_i_1 
       (.I0(\out_tdata[31]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[31]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[31]_INST_0_i_4_n_0 ),
        .O(\packed_data[31]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[32]_i_1 
       (.I0(in_tdata[32]),
        .I1(state__0[0]),
        .I2(in14[32]),
        .I3(state__0[2]),
        .I4(\out_tdata[32]_INST_0_i_2_n_0 ),
        .O(\packed_data[32]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[33]_i_1 
       (.I0(in_tdata[33]),
        .I1(state__0[0]),
        .I2(in14[33]),
        .I3(state__0[2]),
        .I4(\out_tdata[33]_INST_0_i_2_n_0 ),
        .O(\packed_data[33]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[34]_i_1 
       (.I0(in_tdata[34]),
        .I1(state__0[0]),
        .I2(in14[34]),
        .I3(state__0[2]),
        .I4(\out_tdata[34]_INST_0_i_2_n_0 ),
        .O(\packed_data[34]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[35]_i_1 
       (.I0(in_tdata[35]),
        .I1(state__0[0]),
        .I2(in14[35]),
        .I3(state__0[2]),
        .I4(\out_tdata[35]_INST_0_i_2_n_0 ),
        .O(\packed_data[35]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[36]_i_1 
       (.I0(in_tdata[36]),
        .I1(state__0[0]),
        .I2(in14[36]),
        .I3(state__0[2]),
        .I4(\out_tdata[36]_INST_0_i_2_n_0 ),
        .O(\packed_data[36]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[37]_i_1 
       (.I0(in_tdata[37]),
        .I1(state__0[0]),
        .I2(in14[37]),
        .I3(state__0[2]),
        .I4(\out_tdata[37]_INST_0_i_2_n_0 ),
        .O(\packed_data[37]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[38]_i_1 
       (.I0(in_tdata[38]),
        .I1(state__0[0]),
        .I2(in14[38]),
        .I3(state__0[2]),
        .I4(\out_tdata[38]_INST_0_i_2_n_0 ),
        .O(\packed_data[38]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[39]_i_1 
       (.I0(in_tdata[39]),
        .I1(state__0[0]),
        .I2(in14[39]),
        .I3(state__0[2]),
        .I4(\out_tdata[39]_INST_0_i_2_n_0 ),
        .O(\packed_data[39]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[3]_i_1 
       (.I0(\out_tdata[3]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[3]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[3]_INST_0_i_4_n_0 ),
        .O(\packed_data[3]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[40]_i_1 
       (.I0(\out_tdata[40]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[40]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[40]_INST_0_i_4_n_0 ),
        .O(\packed_data[40]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[41]_i_1 
       (.I0(\out_tdata[41]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[41]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[41]_INST_0_i_4_n_0 ),
        .O(\packed_data[41]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[42]_i_1 
       (.I0(\out_tdata[42]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[42]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[42]_INST_0_i_4_n_0 ),
        .O(\packed_data[42]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[43]_i_1 
       (.I0(\out_tdata[43]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[43]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[43]_INST_0_i_4_n_0 ),
        .O(\packed_data[43]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[44]_i_1 
       (.I0(\out_tdata[44]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[44]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[44]_INST_0_i_4_n_0 ),
        .O(\packed_data[44]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[45]_i_1 
       (.I0(\out_tdata[45]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[45]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[45]_INST_0_i_4_n_0 ),
        .O(\packed_data[45]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[46]_i_1 
       (.I0(\out_tdata[46]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[46]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[46]_INST_0_i_4_n_0 ),
        .O(\packed_data[46]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[47]_i_1 
       (.I0(\out_tdata[47]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[47]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[47]_INST_0_i_4_n_0 ),
        .O(\packed_data[47]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[48]_i_1 
       (.I0(in_tdata[48]),
        .I1(state__0[0]),
        .I2(in14[48]),
        .I3(state__0[2]),
        .I4(\out_tdata[48]_INST_0_i_2_n_0 ),
        .O(\packed_data[48]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[49]_i_1 
       (.I0(in_tdata[49]),
        .I1(state__0[0]),
        .I2(in14[49]),
        .I3(state__0[2]),
        .I4(\out_tdata[49]_INST_0_i_2_n_0 ),
        .O(\packed_data[49]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[4]_i_1 
       (.I0(\out_tdata[4]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[4]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[4]_INST_0_i_4_n_0 ),
        .O(\packed_data[4]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[50]_i_1 
       (.I0(in_tdata[50]),
        .I1(state__0[0]),
        .I2(in14[50]),
        .I3(state__0[2]),
        .I4(\out_tdata[50]_INST_0_i_2_n_0 ),
        .O(\packed_data[50]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[51]_i_1 
       (.I0(in_tdata[51]),
        .I1(state__0[0]),
        .I2(in14[51]),
        .I3(state__0[2]),
        .I4(\out_tdata[51]_INST_0_i_2_n_0 ),
        .O(\packed_data[51]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[52]_i_1 
       (.I0(in_tdata[52]),
        .I1(state__0[0]),
        .I2(in14[52]),
        .I3(state__0[2]),
        .I4(\out_tdata[52]_INST_0_i_2_n_0 ),
        .O(\packed_data[52]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[53]_i_1 
       (.I0(in_tdata[53]),
        .I1(state__0[0]),
        .I2(in14[53]),
        .I3(state__0[2]),
        .I4(\out_tdata[53]_INST_0_i_2_n_0 ),
        .O(\packed_data[53]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[54]_i_1 
       (.I0(in_tdata[54]),
        .I1(state__0[0]),
        .I2(in14[54]),
        .I3(state__0[2]),
        .I4(\out_tdata[54]_INST_0_i_2_n_0 ),
        .O(\packed_data[54]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[55]_i_1 
       (.I0(in_tdata[55]),
        .I1(state__0[0]),
        .I2(in14[55]),
        .I3(state__0[2]),
        .I4(\out_tdata[55]_INST_0_i_2_n_0 ),
        .O(\packed_data[55]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[56]_i_1 
       (.I0(in_tdata[56]),
        .I1(state__0[0]),
        .I2(in14[56]),
        .I3(state__0[2]),
        .I4(\out_tdata[56]_INST_0_i_2_n_0 ),
        .O(\packed_data[56]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[57]_i_1 
       (.I0(in_tdata[57]),
        .I1(state__0[0]),
        .I2(in14[57]),
        .I3(state__0[2]),
        .I4(\out_tdata[57]_INST_0_i_2_n_0 ),
        .O(\packed_data[57]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[58]_i_1 
       (.I0(in_tdata[58]),
        .I1(state__0[0]),
        .I2(in14[58]),
        .I3(state__0[2]),
        .I4(\out_tdata[58]_INST_0_i_2_n_0 ),
        .O(\packed_data[58]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[59]_i_1 
       (.I0(in_tdata[59]),
        .I1(state__0[0]),
        .I2(in14[59]),
        .I3(state__0[2]),
        .I4(\out_tdata[59]_INST_0_i_2_n_0 ),
        .O(\packed_data[59]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[5]_i_1 
       (.I0(\out_tdata[5]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[5]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[5]_INST_0_i_4_n_0 ),
        .O(\packed_data[5]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[60]_i_1 
       (.I0(in_tdata[60]),
        .I1(state__0[0]),
        .I2(in14[60]),
        .I3(state__0[2]),
        .I4(\out_tdata[60]_INST_0_i_2_n_0 ),
        .O(\packed_data[60]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[61]_i_1 
       (.I0(in_tdata[61]),
        .I1(state__0[0]),
        .I2(in14[61]),
        .I3(state__0[2]),
        .I4(\out_tdata[61]_INST_0_i_2_n_0 ),
        .O(\packed_data[61]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[62]_i_1 
       (.I0(in_tdata[62]),
        .I1(state__0[0]),
        .I2(in14[62]),
        .I3(state__0[2]),
        .I4(\out_tdata[62]_INST_0_i_2_n_0 ),
        .O(\packed_data[62]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    \packed_data[63]_i_1 
       (.I0(in_tdata[63]),
        .I1(state__0[0]),
        .I2(in14[63]),
        .I3(state__0[2]),
        .I4(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(\packed_data[63]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[6]_i_1 
       (.I0(\out_tdata[6]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[6]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[6]_INST_0_i_4_n_0 ),
        .O(\packed_data[6]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[7]_i_1 
       (.I0(\out_tdata[7]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[7]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[7]_INST_0_i_4_n_0 ),
        .O(\packed_data[7]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[8]_i_1 
       (.I0(\out_tdata[8]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[8]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[8]_INST_0_i_4_n_0 ),
        .O(\packed_data[8]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hB8BBB888)) 
    \packed_data[9]_i_1 
       (.I0(\out_tdata[9]_INST_0_i_1_n_0 ),
        .I1(state__0[2]),
        .I2(\out_tdata[9]_INST_0_i_5_n_0 ),
        .I3(state__0[1]),
        .I4(\out_tdata[9]_INST_0_i_4_n_0 ),
        .O(\packed_data[9]_i_1_n_0 ));
  FDRE \packed_data_reg[0] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[0]_i_1_n_0 ),
        .Q(p_1_in__0[0]),
        .R(1'b0));
  FDRE \packed_data_reg[10] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[10]_i_1_n_0 ),
        .Q(p_1_in__0[10]),
        .R(1'b0));
  FDRE \packed_data_reg[11] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[11]_i_1_n_0 ),
        .Q(p_1_in__0[11]),
        .R(1'b0));
  FDRE \packed_data_reg[12] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[12]_i_1_n_0 ),
        .Q(p_1_in__0[12]),
        .R(1'b0));
  FDRE \packed_data_reg[13] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[13]_i_1_n_0 ),
        .Q(p_1_in__0[13]),
        .R(1'b0));
  FDRE \packed_data_reg[14] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[14]_i_1_n_0 ),
        .Q(p_1_in__0[14]),
        .R(1'b0));
  FDRE \packed_data_reg[15] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[15]_i_1_n_0 ),
        .Q(p_1_in__0[15]),
        .R(1'b0));
  FDRE \packed_data_reg[16] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[16]_i_1_n_0 ),
        .Q(p_1_in__0[16]),
        .R(1'b0));
  FDRE \packed_data_reg[17] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[17]_i_1_n_0 ),
        .Q(p_1_in__0[17]),
        .R(1'b0));
  FDRE \packed_data_reg[18] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[18]_i_1_n_0 ),
        .Q(p_1_in__0[18]),
        .R(1'b0));
  FDRE \packed_data_reg[19] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[19]_i_1_n_0 ),
        .Q(p_1_in__0[19]),
        .R(1'b0));
  FDRE \packed_data_reg[1] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[1]_i_1_n_0 ),
        .Q(p_1_in__0[1]),
        .R(1'b0));
  FDRE \packed_data_reg[20] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[20]_i_1_n_0 ),
        .Q(p_1_in__0[20]),
        .R(1'b0));
  FDRE \packed_data_reg[21] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[21]_i_1_n_0 ),
        .Q(p_1_in__0[21]),
        .R(1'b0));
  FDRE \packed_data_reg[22] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[22]_i_1_n_0 ),
        .Q(p_1_in__0[22]),
        .R(1'b0));
  FDRE \packed_data_reg[23] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[23]_i_1_n_0 ),
        .Q(p_1_in__0[23]),
        .R(1'b0));
  FDRE \packed_data_reg[24] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[24]_i_1_n_0 ),
        .Q(p_1_in__0[24]),
        .R(1'b0));
  FDRE \packed_data_reg[25] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[25]_i_1_n_0 ),
        .Q(p_1_in__0[25]),
        .R(1'b0));
  FDRE \packed_data_reg[26] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[26]_i_1_n_0 ),
        .Q(p_1_in__0[26]),
        .R(1'b0));
  FDRE \packed_data_reg[27] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[27]_i_1_n_0 ),
        .Q(p_1_in__0[27]),
        .R(1'b0));
  FDRE \packed_data_reg[28] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[28]_i_1_n_0 ),
        .Q(p_1_in__0[28]),
        .R(1'b0));
  FDRE \packed_data_reg[29] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[29]_i_1_n_0 ),
        .Q(p_1_in__0[29]),
        .R(1'b0));
  FDRE \packed_data_reg[2] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[2]_i_1_n_0 ),
        .Q(p_1_in__0[2]),
        .R(1'b0));
  FDRE \packed_data_reg[30] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[30]_i_1_n_0 ),
        .Q(p_1_in__0[30]),
        .R(1'b0));
  FDRE \packed_data_reg[31] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[31]_i_1_n_0 ),
        .Q(p_1_in__0[31]),
        .R(1'b0));
  FDRE \packed_data_reg[32] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[32]_i_1_n_0 ),
        .Q(p_1_in__0[32]),
        .R(1'b0));
  FDRE \packed_data_reg[33] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[33]_i_1_n_0 ),
        .Q(p_1_in__0[33]),
        .R(1'b0));
  FDRE \packed_data_reg[34] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[34]_i_1_n_0 ),
        .Q(p_1_in__0[34]),
        .R(1'b0));
  FDRE \packed_data_reg[35] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[35]_i_1_n_0 ),
        .Q(p_1_in__0[35]),
        .R(1'b0));
  FDRE \packed_data_reg[36] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[36]_i_1_n_0 ),
        .Q(p_1_in__0[36]),
        .R(1'b0));
  FDRE \packed_data_reg[37] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[37]_i_1_n_0 ),
        .Q(p_1_in__0[37]),
        .R(1'b0));
  FDRE \packed_data_reg[38] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[38]_i_1_n_0 ),
        .Q(p_1_in__0[38]),
        .R(1'b0));
  FDRE \packed_data_reg[39] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[39]_i_1_n_0 ),
        .Q(p_1_in__0[39]),
        .R(1'b0));
  FDRE \packed_data_reg[3] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[3]_i_1_n_0 ),
        .Q(p_1_in__0[3]),
        .R(1'b0));
  FDRE \packed_data_reg[40] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[40]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[40] ),
        .R(1'b0));
  FDRE \packed_data_reg[41] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[41]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[41] ),
        .R(1'b0));
  FDRE \packed_data_reg[42] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[42]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[42] ),
        .R(1'b0));
  FDRE \packed_data_reg[43] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[43]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[43] ),
        .R(1'b0));
  FDRE \packed_data_reg[44] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[44]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[44] ),
        .R(1'b0));
  FDRE \packed_data_reg[45] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[45]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[45] ),
        .R(1'b0));
  FDRE \packed_data_reg[46] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[46]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[46] ),
        .R(1'b0));
  FDRE \packed_data_reg[47] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[47]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[47] ),
        .R(1'b0));
  FDRE \packed_data_reg[48] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[48]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[48] ),
        .R(1'b0));
  FDRE \packed_data_reg[49] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[49]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[49] ),
        .R(1'b0));
  FDRE \packed_data_reg[4] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[4]_i_1_n_0 ),
        .Q(p_1_in__0[4]),
        .R(1'b0));
  FDRE \packed_data_reg[50] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[50]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[50] ),
        .R(1'b0));
  FDRE \packed_data_reg[51] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[51]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[51] ),
        .R(1'b0));
  FDRE \packed_data_reg[52] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[52]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[52] ),
        .R(1'b0));
  FDRE \packed_data_reg[53] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[53]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[53] ),
        .R(1'b0));
  FDRE \packed_data_reg[54] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[54]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[54] ),
        .R(1'b0));
  FDRE \packed_data_reg[55] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[55]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[55] ),
        .R(1'b0));
  FDRE \packed_data_reg[56] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[56]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[56] ),
        .R(1'b0));
  FDRE \packed_data_reg[57] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[57]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[57] ),
        .R(1'b0));
  FDRE \packed_data_reg[58] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[58]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[58] ),
        .R(1'b0));
  FDRE \packed_data_reg[59] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[59]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[59] ),
        .R(1'b0));
  FDRE \packed_data_reg[5] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[5]_i_1_n_0 ),
        .Q(p_1_in__0[5]),
        .R(1'b0));
  FDRE \packed_data_reg[60] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[60]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[60] ),
        .R(1'b0));
  FDRE \packed_data_reg[61] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[61]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[61] ),
        .R(1'b0));
  FDRE \packed_data_reg[62] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[62]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[62] ),
        .R(1'b0));
  FDRE \packed_data_reg[63] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[63]_i_1_n_0 ),
        .Q(\packed_data_reg_n_0_[63] ),
        .R(1'b0));
  FDRE \packed_data_reg[6] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[6]_i_1_n_0 ),
        .Q(p_1_in__0[6]),
        .R(1'b0));
  FDRE \packed_data_reg[7] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[7]_i_1_n_0 ),
        .Q(p_1_in__0[7]),
        .R(1'b0));
  FDRE \packed_data_reg[8] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[8]_i_1_n_0 ),
        .Q(p_1_in__0[8]),
        .R(1'b0));
  FDRE \packed_data_reg[9] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(\packed_data[9]_i_1_n_0 ),
        .Q(p_1_in__0[9]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[10] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[14]),
        .Q(\samp24_last_reg_n_0_[10] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[11] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[15]),
        .Q(\samp24_last_reg_n_0_[11] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[12] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[20]),
        .Q(\samp24_last_reg_n_0_[12] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[13] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[21]),
        .Q(\samp24_last_reg_n_0_[13] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[14] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[22]),
        .Q(\samp24_last_reg_n_0_[14] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[15] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[23]),
        .Q(\samp24_last_reg_n_0_[15] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[16] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[24]),
        .Q(data3[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[17] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[25]),
        .Q(data3[1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[18] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[26]),
        .Q(data3[2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[19] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[27]),
        .Q(data3[3]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[20] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[28]),
        .Q(data3[4]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[21] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[29]),
        .Q(data3[5]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[22] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[30]),
        .Q(data3[6]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[23] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[31]),
        .Q(data3[7]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[8] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[12]),
        .Q(\samp24_last_reg_n_0_[8] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp24_last_reg[9] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[13]),
        .Q(\samp24_last_reg_n_0_[9] ),
        .R(1'b0));
  LUT3 #(
    .INIT(8'h20)) 
    \samp48_last[47]_i_1 
       (.I0(in_tvalid),
        .I1(extra_cycle_reg_n_0),
        .I2(out_tready),
        .O(p_2_in15_out));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[24] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[36]),
        .Q(\samp48_last_reg_n_0_[24] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[25] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[37]),
        .Q(\samp48_last_reg_n_0_[25] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[26] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[38]),
        .Q(\samp48_last_reg_n_0_[26] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[27] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[39]),
        .Q(\samp48_last_reg_n_0_[27] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[28] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[40]),
        .Q(\samp48_last_reg_n_0_[28] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[29] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[41]),
        .Q(\samp48_last_reg_n_0_[29] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[30] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[42]),
        .Q(\samp48_last_reg_n_0_[30] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[31] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[43]),
        .Q(\samp48_last_reg_n_0_[31] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[32] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[44]),
        .Q(p_1_in[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[33] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[45]),
        .Q(p_1_in[1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[34] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[46]),
        .Q(p_1_in[2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[35] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[47]),
        .Q(p_1_in[3]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[36] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[52]),
        .Q(p_1_in[4]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[37] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[53]),
        .Q(p_1_in[5]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[38] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[54]),
        .Q(p_1_in[6]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[39] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[55]),
        .Q(p_1_in[7]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[40] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[56]),
        .Q(p_1_in[8]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[41] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[57]),
        .Q(p_1_in[9]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[42] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[58]),
        .Q(p_1_in[10]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[43] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[59]),
        .Q(p_1_in[11]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[44] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[60]),
        .Q(p_1_in[12]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[45] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[61]),
        .Q(p_1_in[13]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[46] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[62]),
        .Q(p_1_in[14]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \samp48_last_reg[47] 
       (.C(clk),
        .CE(p_2_in15_out),
        .D(in_tdata[63]),
        .Q(p_1_in[15]),
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
