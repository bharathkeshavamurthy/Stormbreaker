// Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2018.3 (lin64) Build 2405991 Thu Dec  6 23:36:41 MST 2018
// Date        : Wed Nov 18 10:51:25 2020
// Host        : bender.ad.sklk.us running 64-bit Ubuntu 16.04.6 LTS
// Command     : write_verilog -force -mode funcsim std_pulses_sim.v
// Design      : std_pulses
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7z030sbg485-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CLK_PERIOD_PS = "10000" *) 
(* NotValidForBitStream *)
module std_pulses
   (clk,
    rst,
    tick_1us,
    tick_10us,
    tick_100us,
    tick_1ms,
    tick_10ms,
    tick_100ms,
    tick_250ms,
    tick_1s,
    hb_normal,
    hb_error,
    hb_alert,
    breath_mode,
    breath);
  input clk;
  input rst;
  output tick_1us;
  output tick_10us;
  output tick_100us;
  output tick_1ms;
  output tick_10ms;
  output tick_100ms;
  output tick_250ms;
  output tick_1s;
  output hb_normal;
  output hb_error;
  output hb_alert;
  input [2:0]breath_mode;
  output breath;

  wire \blip_r_reg_n_0_[0] ;
  wire \blip_r_reg_n_0_[1] ;
  wire \blip_r_reg_n_0_[2] ;
  wire \blip_r_reg_n_0_[3] ;
  wire breath;
  wire breath_INST_0_i_1_n_0;
  wire breath_INST_0_i_2_n_0;
  wire breath_INST_0_i_3_n_0;
  wire breath_down;
  wire breath_down_i_1_n_0;
  wire breath_down_i_3_n_0;
  wire breath_down_i_4_n_0;
  wire breath_down_reg_n_0;
  wire [2:0]breath_mode;
  wire breath_pwm1;
  wire \breath_pwm[4]_i_3_n_0 ;
  wire \breath_pwm[4]_i_4_n_0 ;
  wire \breath_pwm[4]_i_5_n_0 ;
  wire \breath_pwm[7]_i_2_n_0 ;
  wire \breath_pwm[7]_i_3_n_0 ;
  wire \breath_pwm[7]_i_4_n_0 ;
  wire \breath_pwm_reg[4]_i_1_n_0 ;
  wire \breath_pwm_reg[4]_i_1_n_1 ;
  wire \breath_pwm_reg[4]_i_1_n_2 ;
  wire \breath_pwm_reg[4]_i_1_n_3 ;
  wire \breath_pwm_reg[4]_i_1_n_4 ;
  wire \breath_pwm_reg[4]_i_1_n_5 ;
  wire \breath_pwm_reg[4]_i_1_n_6 ;
  wire \breath_pwm_reg[4]_i_1_n_7 ;
  wire \breath_pwm_reg[7]_i_1_n_2 ;
  wire \breath_pwm_reg[7]_i_1_n_3 ;
  wire \breath_pwm_reg[7]_i_1_n_5 ;
  wire \breath_pwm_reg[7]_i_1_n_6 ;
  wire \breath_pwm_reg[7]_i_1_n_7 ;
  wire [7:1]breath_pwm_reg__0;
  wire clk;
  wire \cnt[0]_i_1_n_0 ;
  wire \cnt[7]_i_2_n_0 ;
  wire \counter[2]_i_1__7_n_0 ;
  wire \counter[3]_i_1__0_n_0 ;
  wire \counter[3]_i_1__1_n_0 ;
  wire \counter[3]_i_1__2_n_0 ;
  wire \counter[3]_i_1__3_n_0 ;
  wire \counter[3]_i_1__4_n_0 ;
  wire \counter[3]_i_1_n_0 ;
  wire \counter[4]_i_1__1_n_0 ;
  wire \counter[4]_i_1_n_0 ;
  wire \counter[6]_i_1_n_0 ;
  wire hb_alert;
  wire hb_error;
  wire \hb_error_counter[2]_i_2_n_0 ;
  wire \hb_error_counter_reg_n_0_[1] ;
  wire hb_normal;
  wire [0:0]hb_normal_counter;
  wire \hb_normal_counter[1]_i_2_n_0 ;
  wire i__carry_i_1_n_0;
  wire i__carry_i_2_n_0;
  wire i__carry_i_3_n_0;
  wire i__carry_i_4_n_0;
  wire i__carry_i_5_n_0;
  wire i__carry_i_6_n_0;
  wire i__carry_i_7_n_0;
  wire i__carry_i_8_n_0;
  wire out_carry_i_1_n_0;
  wire out_carry_i_2_n_0;
  wire [1:0]p_0_in;
  wire [6:0]p_0_in__0;
  wire [3:0]p_0_in__1;
  wire [2:0]p_0_in__10;
  wire [3:0]p_0_in__2;
  wire [3:0]p_0_in__3;
  wire [3:0]p_0_in__4;
  wire [3:0]p_0_in__5;
  wire [4:0]p_0_in__6;
  wire [3:0]p_0_in__7;
  wire [4:0]p_0_in__8;
  wire [7:1]p_0_in__9;
  wire rst;
  wire tick_100ms;
  wire tick_100us;
  wire tick_100us_INST_0_i_1_n_0;
  wire tick_10ms;
  wire tick_10ms_INST_0_i_1_n_0;
  wire tick_10us;
  wire tick_10us_INST_0_i_1_n_0;
  wire tick_10us_INST_0_i_2_n_0;
  wire tick_1ms;
  wire tick_1ms_INST_0_i_1_n_0;
  wire tick_1ms_INST_0_i_2_n_0;
  wire tick_1s;
  wire tick_1s_INST_0_i_1_n_0;
  wire tick_1us;
  wire tick_1us_INST_0_i_1_n_0;
  wire tick_250ms;
  wire [3:0]\u_100ms_timer/counter_reg__0 ;
  wire [3:0]\u_100us_timer/counter_reg__0 ;
  wire [3:0]\u_10ms_timer/counter_reg__0 ;
  wire [3:0]\u_10us_timer/counter_reg__0 ;
  wire [3:0]\u_1ms_timer/counter_reg__0 ;
  wire [3:0]\u_1s_timer/counter_reg__0 ;
  wire [4:0]\u_250ms_timer/counter_reg__0 ;
  wire [6:0]\u_base_timer/counter_reg__0 ;
  wire [7:0]\u_breath/cnt_reg__0 ;
  wire \u_breath/pwm0_inferred__0/i__carry_n_0 ;
  wire \u_breath/pwm0_inferred__0/i__carry_n_1 ;
  wire \u_breath/pwm0_inferred__0/i__carry_n_2 ;
  wire \u_breath/pwm0_inferred__0/i__carry_n_3 ;
  wire [4:0]\u_breath_tick/counter_reg__0 ;
  wire \u_breath_tick/out_carry_n_2 ;
  wire \u_breath_tick/out_carry_n_3 ;
  wire [3:2]\NLW_breath_pwm_reg[7]_i_1_CO_UNCONNECTED ;
  wire [3:3]\NLW_breath_pwm_reg[7]_i_1_O_UNCONNECTED ;
  wire [3:0]\NLW_u_breath/pwm0_inferred__0/i__carry_O_UNCONNECTED ;
  wire [3:2]\NLW_u_breath_tick/out_carry_CO_UNCONNECTED ;
  wire [3:0]\NLW_u_breath_tick/out_carry_O_UNCONNECTED ;

  FDSE \blip_r_reg[0] 
       (.C(clk),
        .CE(tick_1s),
        .D(\blip_r_reg_n_0_[3] ),
        .Q(\blip_r_reg_n_0_[0] ),
        .S(rst));
  FDRE \blip_r_reg[1] 
       (.C(clk),
        .CE(tick_1s),
        .D(\blip_r_reg_n_0_[0] ),
        .Q(\blip_r_reg_n_0_[1] ),
        .R(rst));
  FDRE \blip_r_reg[2] 
       (.C(clk),
        .CE(tick_1s),
        .D(\blip_r_reg_n_0_[1] ),
        .Q(\blip_r_reg_n_0_[2] ),
        .R(rst));
  FDRE \blip_r_reg[3] 
       (.C(clk),
        .CE(tick_1s),
        .D(\blip_r_reg_n_0_[2] ),
        .Q(\blip_r_reg_n_0_[3] ),
        .R(rst));
  LUT6 #(
    .INIT(64'h80008000BFFF8000)) 
    breath_INST_0
       (.I0(hb_alert),
        .I1(breath_INST_0_i_1_n_0),
        .I2(\blip_r_reg_n_0_[0] ),
        .I3(breath_mode[0]),
        .I4(\u_breath/pwm0_inferred__0/i__carry_n_0 ),
        .I5(breath_INST_0_i_2_n_0),
        .O(breath));
  LUT2 #(
    .INIT(4'h2)) 
    breath_INST_0_i_1
       (.I0(breath_mode[2]),
        .I1(breath_mode[1]),
        .O(breath_INST_0_i_1_n_0));
  LUT5 #(
    .INIT(32'h00000002)) 
    breath_INST_0_i_2
       (.I0(breath_INST_0_i_3_n_0),
        .I1(breath_pwm_reg__0[7]),
        .I2(breath_pwm_reg__0[1]),
        .I3(breath_pwm_reg__0[2]),
        .I4(breath_pwm_reg__0[3]),
        .O(breath_INST_0_i_2_n_0));
  LUT3 #(
    .INIT(8'h01)) 
    breath_INST_0_i_3
       (.I0(breath_pwm_reg__0[6]),
        .I1(breath_pwm_reg__0[5]),
        .I2(breath_pwm_reg__0[4]),
        .O(breath_INST_0_i_3_n_0));
  LUT6 #(
    .INIT(64'hFFFFBFFE00008002)) 
    breath_down_i_1
       (.I0(breath_down),
        .I1(breath_pwm_reg__0[3]),
        .I2(breath_pwm_reg__0[2]),
        .I3(breath_pwm_reg__0[4]),
        .I4(breath_down_i_3_n_0),
        .I5(breath_down_reg_n_0),
        .O(breath_down_i_1_n_0));
  LUT5 #(
    .INIT(32'h08000000)) 
    breath_down_i_2
       (.I0(breath_down_i_4_n_0),
        .I1(breath_pwm_reg__0[7]),
        .I2(breath_pwm_reg__0[1]),
        .I3(breath_pwm_reg__0[2]),
        .I4(breath_pwm_reg__0[3]),
        .O(breath_down));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT5 #(
    .INIT(32'hFFFF7FFE)) 
    breath_down_i_3
       (.I0(breath_pwm_reg__0[5]),
        .I1(breath_pwm_reg__0[4]),
        .I2(breath_pwm_reg__0[6]),
        .I3(breath_pwm_reg__0[7]),
        .I4(breath_pwm_reg__0[1]),
        .O(breath_down_i_3_n_0));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT3 #(
    .INIT(8'h80)) 
    breath_down_i_4
       (.I0(breath_pwm_reg__0[5]),
        .I1(breath_pwm_reg__0[6]),
        .I2(breath_pwm_reg__0[4]),
        .O(breath_down_i_4_n_0));
  FDRE breath_down_reg
       (.C(clk),
        .CE(1'b1),
        .D(breath_down_i_1_n_0),
        .Q(breath_down_reg_n_0),
        .R(rst));
  LUT2 #(
    .INIT(4'h2)) 
    \breath_pwm[4]_i_2 
       (.I0(\u_breath_tick/out_carry_n_2 ),
        .I1(breath_down_reg_n_0),
        .O(breath_pwm1));
  LUT3 #(
    .INIT(8'h4B)) 
    \breath_pwm[4]_i_3 
       (.I0(breath_down_reg_n_0),
        .I1(\u_breath_tick/out_carry_n_2 ),
        .I2(breath_pwm_reg__0[4]),
        .O(\breath_pwm[4]_i_3_n_0 ));
  LUT3 #(
    .INIT(8'h4B)) 
    \breath_pwm[4]_i_4 
       (.I0(breath_down_reg_n_0),
        .I1(\u_breath_tick/out_carry_n_2 ),
        .I2(breath_pwm_reg__0[3]),
        .O(\breath_pwm[4]_i_4_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \breath_pwm[4]_i_5 
       (.I0(breath_pwm_reg__0[2]),
        .O(\breath_pwm[4]_i_5_n_0 ));
  LUT2 #(
    .INIT(4'h9)) 
    \breath_pwm[7]_i_2 
       (.I0(breath_pwm_reg__0[6]),
        .I1(breath_pwm_reg__0[7]),
        .O(\breath_pwm[7]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h9)) 
    \breath_pwm[7]_i_3 
       (.I0(breath_pwm_reg__0[5]),
        .I1(breath_pwm_reg__0[6]),
        .O(\breath_pwm[7]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h9)) 
    \breath_pwm[7]_i_4 
       (.I0(breath_pwm_reg__0[4]),
        .I1(breath_pwm_reg__0[5]),
        .O(\breath_pwm[7]_i_4_n_0 ));
  FDRE \breath_pwm_reg[1] 
       (.C(clk),
        .CE(\u_breath_tick/out_carry_n_2 ),
        .D(\breath_pwm_reg[4]_i_1_n_7 ),
        .Q(breath_pwm_reg__0[1]),
        .R(rst));
  FDRE \breath_pwm_reg[2] 
       (.C(clk),
        .CE(\u_breath_tick/out_carry_n_2 ),
        .D(\breath_pwm_reg[4]_i_1_n_6 ),
        .Q(breath_pwm_reg__0[2]),
        .R(rst));
  FDRE \breath_pwm_reg[3] 
       (.C(clk),
        .CE(\u_breath_tick/out_carry_n_2 ),
        .D(\breath_pwm_reg[4]_i_1_n_5 ),
        .Q(breath_pwm_reg__0[3]),
        .R(rst));
  FDRE \breath_pwm_reg[4] 
       (.C(clk),
        .CE(\u_breath_tick/out_carry_n_2 ),
        .D(\breath_pwm_reg[4]_i_1_n_4 ),
        .Q(breath_pwm_reg__0[4]),
        .R(rst));
  CARRY4 \breath_pwm_reg[4]_i_1 
       (.CI(1'b0),
        .CO({\breath_pwm_reg[4]_i_1_n_0 ,\breath_pwm_reg[4]_i_1_n_1 ,\breath_pwm_reg[4]_i_1_n_2 ,\breath_pwm_reg[4]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({breath_pwm1,breath_pwm_reg__0[3:2],1'b0}),
        .O({\breath_pwm_reg[4]_i_1_n_4 ,\breath_pwm_reg[4]_i_1_n_5 ,\breath_pwm_reg[4]_i_1_n_6 ,\breath_pwm_reg[4]_i_1_n_7 }),
        .S({\breath_pwm[4]_i_3_n_0 ,\breath_pwm[4]_i_4_n_0 ,\breath_pwm[4]_i_5_n_0 ,breath_pwm_reg__0[1]}));
  FDRE \breath_pwm_reg[5] 
       (.C(clk),
        .CE(\u_breath_tick/out_carry_n_2 ),
        .D(\breath_pwm_reg[7]_i_1_n_7 ),
        .Q(breath_pwm_reg__0[5]),
        .R(rst));
  FDRE \breath_pwm_reg[6] 
       (.C(clk),
        .CE(\u_breath_tick/out_carry_n_2 ),
        .D(\breath_pwm_reg[7]_i_1_n_6 ),
        .Q(breath_pwm_reg__0[6]),
        .R(rst));
  FDRE \breath_pwm_reg[7] 
       (.C(clk),
        .CE(\u_breath_tick/out_carry_n_2 ),
        .D(\breath_pwm_reg[7]_i_1_n_5 ),
        .Q(breath_pwm_reg__0[7]),
        .R(rst));
  CARRY4 \breath_pwm_reg[7]_i_1 
       (.CI(\breath_pwm_reg[4]_i_1_n_0 ),
        .CO({\NLW_breath_pwm_reg[7]_i_1_CO_UNCONNECTED [3:2],\breath_pwm_reg[7]_i_1_n_2 ,\breath_pwm_reg[7]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,breath_pwm_reg__0[5:4]}),
        .O({\NLW_breath_pwm_reg[7]_i_1_O_UNCONNECTED [3],\breath_pwm_reg[7]_i_1_n_5 ,\breath_pwm_reg[7]_i_1_n_6 ,\breath_pwm_reg[7]_i_1_n_7 }),
        .S({1'b0,\breath_pwm[7]_i_2_n_0 ,\breath_pwm[7]_i_3_n_0 ,\breath_pwm[7]_i_4_n_0 }));
  LUT1 #(
    .INIT(2'h1)) 
    \cnt[0]_i_1 
       (.I0(\u_breath/cnt_reg__0 [0]),
        .O(\cnt[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \cnt[1]_i_1 
       (.I0(\u_breath/cnt_reg__0 [1]),
        .I1(\u_breath/cnt_reg__0 [0]),
        .O(p_0_in__9[1]));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \cnt[2]_i_1 
       (.I0(\u_breath/cnt_reg__0 [2]),
        .I1(\u_breath/cnt_reg__0 [1]),
        .I2(\u_breath/cnt_reg__0 [0]),
        .O(p_0_in__9[2]));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \cnt[3]_i_1 
       (.I0(\u_breath/cnt_reg__0 [3]),
        .I1(\u_breath/cnt_reg__0 [0]),
        .I2(\u_breath/cnt_reg__0 [1]),
        .I3(\u_breath/cnt_reg__0 [2]),
        .O(p_0_in__9[3]));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT5 #(
    .INIT(32'h6AAAAAAA)) 
    \cnt[4]_i_1 
       (.I0(\u_breath/cnt_reg__0 [4]),
        .I1(\u_breath/cnt_reg__0 [2]),
        .I2(\u_breath/cnt_reg__0 [1]),
        .I3(\u_breath/cnt_reg__0 [0]),
        .I4(\u_breath/cnt_reg__0 [3]),
        .O(p_0_in__9[4]));
  LUT6 #(
    .INIT(64'h6AAAAAAAAAAAAAAA)) 
    \cnt[5]_i_1 
       (.I0(\u_breath/cnt_reg__0 [5]),
        .I1(\u_breath/cnt_reg__0 [3]),
        .I2(\u_breath/cnt_reg__0 [0]),
        .I3(\u_breath/cnt_reg__0 [1]),
        .I4(\u_breath/cnt_reg__0 [2]),
        .I5(\u_breath/cnt_reg__0 [4]),
        .O(p_0_in__9[5]));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT2 #(
    .INIT(4'h9)) 
    \cnt[6]_i_1 
       (.I0(\cnt[7]_i_2_n_0 ),
        .I1(\u_breath/cnt_reg__0 [6]),
        .O(p_0_in__9[6]));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT3 #(
    .INIT(8'h9A)) 
    \cnt[7]_i_1 
       (.I0(\u_breath/cnt_reg__0 [7]),
        .I1(\cnt[7]_i_2_n_0 ),
        .I2(\u_breath/cnt_reg__0 [6]),
        .O(p_0_in__9[7]));
  LUT6 #(
    .INIT(64'h7FFFFFFFFFFFFFFF)) 
    \cnt[7]_i_2 
       (.I0(\u_breath/cnt_reg__0 [4]),
        .I1(\u_breath/cnt_reg__0 [2]),
        .I2(\u_breath/cnt_reg__0 [1]),
        .I3(\u_breath/cnt_reg__0 [0]),
        .I4(\u_breath/cnt_reg__0 [3]),
        .I5(\u_breath/cnt_reg__0 [5]),
        .O(\cnt[7]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT1 #(
    .INIT(2'h1)) 
    \counter[0]_i_1 
       (.I0(\u_base_timer/counter_reg__0 [0]),
        .O(p_0_in__0[0]));
  LUT1 #(
    .INIT(2'h1)) 
    \counter[0]_i_1__0 
       (.I0(\u_10us_timer/counter_reg__0 [0]),
        .O(p_0_in__1[0]));
  LUT1 #(
    .INIT(2'h1)) 
    \counter[0]_i_1__1 
       (.I0(\u_100us_timer/counter_reg__0 [0]),
        .O(p_0_in__2[0]));
  LUT1 #(
    .INIT(2'h1)) 
    \counter[0]_i_1__2 
       (.I0(\u_1ms_timer/counter_reg__0 [0]),
        .O(p_0_in__3[0]));
  LUT1 #(
    .INIT(2'h1)) 
    \counter[0]_i_1__3 
       (.I0(\u_10ms_timer/counter_reg__0 [0]),
        .O(p_0_in__4[0]));
  LUT1 #(
    .INIT(2'h1)) 
    \counter[0]_i_1__4 
       (.I0(\u_100ms_timer/counter_reg__0 [0]),
        .O(p_0_in__5[0]));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT1 #(
    .INIT(2'h1)) 
    \counter[0]_i_1__5 
       (.I0(\u_250ms_timer/counter_reg__0 [0]),
        .O(p_0_in__6[0]));
  LUT1 #(
    .INIT(2'h1)) 
    \counter[0]_i_1__6 
       (.I0(\u_1s_timer/counter_reg__0 [0]),
        .O(p_0_in__7[0]));
  LUT1 #(
    .INIT(2'h1)) 
    \counter[0]_i_1__7 
       (.I0(\u_breath_tick/counter_reg__0 [0]),
        .O(p_0_in__8[0]));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counter[1]_i_1 
       (.I0(\u_base_timer/counter_reg__0 [0]),
        .I1(\u_base_timer/counter_reg__0 [1]),
        .O(p_0_in__0[1]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counter[1]_i_1__0 
       (.I0(\u_10us_timer/counter_reg__0 [0]),
        .I1(\u_10us_timer/counter_reg__0 [1]),
        .O(p_0_in__1[1]));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counter[1]_i_1__1 
       (.I0(\u_100us_timer/counter_reg__0 [0]),
        .I1(\u_100us_timer/counter_reg__0 [1]),
        .O(p_0_in__2[1]));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counter[1]_i_1__2 
       (.I0(\u_1ms_timer/counter_reg__0 [0]),
        .I1(\u_1ms_timer/counter_reg__0 [1]),
        .O(p_0_in__3[1]));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counter[1]_i_1__3 
       (.I0(\u_10ms_timer/counter_reg__0 [0]),
        .I1(\u_10ms_timer/counter_reg__0 [1]),
        .O(p_0_in__4[1]));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counter[1]_i_1__4 
       (.I0(\u_100ms_timer/counter_reg__0 [0]),
        .I1(\u_100ms_timer/counter_reg__0 [1]),
        .O(p_0_in__5[1]));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counter[1]_i_1__5 
       (.I0(\u_250ms_timer/counter_reg__0 [0]),
        .I1(\u_250ms_timer/counter_reg__0 [1]),
        .O(p_0_in__6[1]));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counter[1]_i_1__6 
       (.I0(\u_1s_timer/counter_reg__0 [0]),
        .I1(\u_1s_timer/counter_reg__0 [1]),
        .O(p_0_in__7[1]));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \counter[1]_i_1__7 
       (.I0(\u_breath_tick/counter_reg__0 [0]),
        .I1(\u_breath_tick/counter_reg__0 [1]),
        .O(p_0_in__8[1]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \counter[2]_i_1 
       (.I0(\u_10us_timer/counter_reg__0 [2]),
        .I1(\u_10us_timer/counter_reg__0 [1]),
        .I2(\u_10us_timer/counter_reg__0 [0]),
        .O(p_0_in__1[2]));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \counter[2]_i_1__0 
       (.I0(\u_100us_timer/counter_reg__0 [2]),
        .I1(\u_100us_timer/counter_reg__0 [1]),
        .I2(\u_100us_timer/counter_reg__0 [0]),
        .O(p_0_in__2[2]));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \counter[2]_i_1__1 
       (.I0(\u_1ms_timer/counter_reg__0 [2]),
        .I1(\u_1ms_timer/counter_reg__0 [1]),
        .I2(\u_1ms_timer/counter_reg__0 [0]),
        .O(p_0_in__3[2]));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \counter[2]_i_1__2 
       (.I0(\u_10ms_timer/counter_reg__0 [2]),
        .I1(\u_10ms_timer/counter_reg__0 [1]),
        .I2(\u_10ms_timer/counter_reg__0 [0]),
        .O(p_0_in__4[2]));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \counter[2]_i_1__3 
       (.I0(\u_100ms_timer/counter_reg__0 [2]),
        .I1(\u_100ms_timer/counter_reg__0 [1]),
        .I2(\u_100ms_timer/counter_reg__0 [0]),
        .O(p_0_in__5[2]));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \counter[2]_i_1__4 
       (.I0(\u_250ms_timer/counter_reg__0 [2]),
        .I1(\u_250ms_timer/counter_reg__0 [1]),
        .I2(\u_250ms_timer/counter_reg__0 [0]),
        .O(p_0_in__6[2]));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \counter[2]_i_1__5 
       (.I0(\u_1s_timer/counter_reg__0 [2]),
        .I1(\u_1s_timer/counter_reg__0 [1]),
        .I2(\u_1s_timer/counter_reg__0 [0]),
        .O(p_0_in__7[2]));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \counter[2]_i_1__6 
       (.I0(\u_breath_tick/counter_reg__0 [2]),
        .I1(\u_breath_tick/counter_reg__0 [1]),
        .I2(\u_breath_tick/counter_reg__0 [0]),
        .O(p_0_in__8[2]));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \counter[2]_i_1__7 
       (.I0(\u_base_timer/counter_reg__0 [2]),
        .I1(\u_base_timer/counter_reg__0 [1]),
        .I2(\u_base_timer/counter_reg__0 [0]),
        .O(\counter[2]_i_1__7_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAAEAAAAAAAAAA)) 
    \counter[3]_i_1 
       (.I0(rst),
        .I1(\u_10us_timer/counter_reg__0 [0]),
        .I2(\u_10us_timer/counter_reg__0 [1]),
        .I3(\u_10us_timer/counter_reg__0 [3]),
        .I4(\u_10us_timer/counter_reg__0 [2]),
        .I5(tick_1us),
        .O(\counter[3]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAAEAAAAAAAAAA)) 
    \counter[3]_i_1__0 
       (.I0(rst),
        .I1(\u_100us_timer/counter_reg__0 [0]),
        .I2(\u_100us_timer/counter_reg__0 [1]),
        .I3(\u_100us_timer/counter_reg__0 [3]),
        .I4(\u_100us_timer/counter_reg__0 [2]),
        .I5(tick_10us),
        .O(\counter[3]_i_1__0_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \counter[3]_i_1__1 
       (.I0(rst),
        .I1(tick_1ms),
        .O(\counter[3]_i_1__1_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAAEAAAAAAAAAA)) 
    \counter[3]_i_1__2 
       (.I0(rst),
        .I1(\u_10ms_timer/counter_reg__0 [0]),
        .I2(\u_10ms_timer/counter_reg__0 [1]),
        .I3(\u_10ms_timer/counter_reg__0 [3]),
        .I4(\u_10ms_timer/counter_reg__0 [2]),
        .I5(tick_1ms),
        .O(\counter[3]_i_1__2_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAAEAAAAAAAAAA)) 
    \counter[3]_i_1__3 
       (.I0(rst),
        .I1(\u_100ms_timer/counter_reg__0 [0]),
        .I2(\u_100ms_timer/counter_reg__0 [1]),
        .I3(\u_100ms_timer/counter_reg__0 [3]),
        .I4(\u_100ms_timer/counter_reg__0 [2]),
        .I5(tick_10ms),
        .O(\counter[3]_i_1__3_n_0 ));
  LUT4 #(
    .INIT(16'hAABA)) 
    \counter[3]_i_1__4 
       (.I0(rst),
        .I1(tick_1s_INST_0_i_1_n_0),
        .I2(tick_10ms),
        .I3(\hb_error_counter[2]_i_2_n_0 ),
        .O(\counter[3]_i_1__4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \counter[3]_i_1__5 
       (.I0(\u_base_timer/counter_reg__0 [3]),
        .I1(\u_base_timer/counter_reg__0 [0]),
        .I2(\u_base_timer/counter_reg__0 [1]),
        .I3(\u_base_timer/counter_reg__0 [2]),
        .O(p_0_in__0[3]));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \counter[3]_i_1__6 
       (.I0(\u_250ms_timer/counter_reg__0 [3]),
        .I1(\u_250ms_timer/counter_reg__0 [0]),
        .I2(\u_250ms_timer/counter_reg__0 [1]),
        .I3(\u_250ms_timer/counter_reg__0 [2]),
        .O(p_0_in__6[3]));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \counter[3]_i_1__7 
       (.I0(\u_breath_tick/counter_reg__0 [3]),
        .I1(\u_breath_tick/counter_reg__0 [0]),
        .I2(\u_breath_tick/counter_reg__0 [1]),
        .I3(\u_breath_tick/counter_reg__0 [2]),
        .O(p_0_in__8[3]));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \counter[3]_i_2 
       (.I0(\u_10us_timer/counter_reg__0 [3]),
        .I1(\u_10us_timer/counter_reg__0 [0]),
        .I2(\u_10us_timer/counter_reg__0 [1]),
        .I3(\u_10us_timer/counter_reg__0 [2]),
        .O(p_0_in__1[3]));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \counter[3]_i_2__0 
       (.I0(\u_100us_timer/counter_reg__0 [3]),
        .I1(\u_100us_timer/counter_reg__0 [0]),
        .I2(\u_100us_timer/counter_reg__0 [1]),
        .I3(\u_100us_timer/counter_reg__0 [2]),
        .O(p_0_in__2[3]));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \counter[3]_i_2__1 
       (.I0(\u_1ms_timer/counter_reg__0 [3]),
        .I1(\u_1ms_timer/counter_reg__0 [0]),
        .I2(\u_1ms_timer/counter_reg__0 [1]),
        .I3(\u_1ms_timer/counter_reg__0 [2]),
        .O(p_0_in__3[3]));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \counter[3]_i_2__2 
       (.I0(\u_10ms_timer/counter_reg__0 [3]),
        .I1(\u_10ms_timer/counter_reg__0 [0]),
        .I2(\u_10ms_timer/counter_reg__0 [1]),
        .I3(\u_10ms_timer/counter_reg__0 [2]),
        .O(p_0_in__4[3]));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \counter[3]_i_2__3 
       (.I0(\u_100ms_timer/counter_reg__0 [3]),
        .I1(\u_100ms_timer/counter_reg__0 [0]),
        .I2(\u_100ms_timer/counter_reg__0 [1]),
        .I3(\u_100ms_timer/counter_reg__0 [2]),
        .O(p_0_in__5[3]));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \counter[3]_i_2__4 
       (.I0(\u_1s_timer/counter_reg__0 [3]),
        .I1(\u_1s_timer/counter_reg__0 [0]),
        .I2(\u_1s_timer/counter_reg__0 [1]),
        .I3(\u_1s_timer/counter_reg__0 [2]),
        .O(p_0_in__7[3]));
  LUT3 #(
    .INIT(8'hBA)) 
    \counter[4]_i_1 
       (.I0(rst),
        .I1(\hb_normal_counter[1]_i_2_n_0 ),
        .I2(tick_10ms),
        .O(\counter[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT5 #(
    .INIT(32'h6AAAAAAA)) 
    \counter[4]_i_1__0 
       (.I0(\u_base_timer/counter_reg__0 [4]),
        .I1(\u_base_timer/counter_reg__0 [2]),
        .I2(\u_base_timer/counter_reg__0 [1]),
        .I3(\u_base_timer/counter_reg__0 [0]),
        .I4(\u_base_timer/counter_reg__0 [3]),
        .O(p_0_in__0[4]));
  LUT2 #(
    .INIT(4'hE)) 
    \counter[4]_i_1__1 
       (.I0(rst),
        .I1(\u_breath_tick/out_carry_n_2 ),
        .O(\counter[4]_i_1__1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'h6AAAAAAA)) 
    \counter[4]_i_2 
       (.I0(\u_250ms_timer/counter_reg__0 [4]),
        .I1(\u_250ms_timer/counter_reg__0 [2]),
        .I2(\u_250ms_timer/counter_reg__0 [1]),
        .I3(\u_250ms_timer/counter_reg__0 [0]),
        .I4(\u_250ms_timer/counter_reg__0 [3]),
        .O(p_0_in__6[4]));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT5 #(
    .INIT(32'h6AAAAAAA)) 
    \counter[4]_i_2__0 
       (.I0(\u_breath_tick/counter_reg__0 [4]),
        .I1(\u_breath_tick/counter_reg__0 [2]),
        .I2(\u_breath_tick/counter_reg__0 [1]),
        .I3(\u_breath_tick/counter_reg__0 [0]),
        .I4(\u_breath_tick/counter_reg__0 [3]),
        .O(p_0_in__8[4]));
  LUT6 #(
    .INIT(64'h6AAAAAAAAAAAAAAA)) 
    \counter[5]_i_1 
       (.I0(\u_base_timer/counter_reg__0 [5]),
        .I1(\u_base_timer/counter_reg__0 [3]),
        .I2(\u_base_timer/counter_reg__0 [0]),
        .I3(\u_base_timer/counter_reg__0 [1]),
        .I4(\u_base_timer/counter_reg__0 [2]),
        .I5(\u_base_timer/counter_reg__0 [4]),
        .O(p_0_in__0[5]));
  LUT2 #(
    .INIT(4'hE)) 
    \counter[6]_i_1 
       (.I0(rst),
        .I1(tick_1us),
        .O(\counter[6]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAA6AAAAAAAAAAAAA)) 
    \counter[6]_i_2 
       (.I0(\u_base_timer/counter_reg__0 [6]),
        .I1(\u_base_timer/counter_reg__0 [4]),
        .I2(\u_base_timer/counter_reg__0 [2]),
        .I3(tick_1us_INST_0_i_1_n_0),
        .I4(\u_base_timer/counter_reg__0 [3]),
        .I5(\u_base_timer/counter_reg__0 [5]),
        .O(p_0_in__0[6]));
  LUT6 #(
    .INIT(64'hAAAAA6AAAAAAAAAA)) 
    \hb_error_counter[0]_i_1 
       (.I0(hb_alert),
        .I1(\u_100ms_timer/counter_reg__0 [0]),
        .I2(\u_100ms_timer/counter_reg__0 [1]),
        .I3(\u_100ms_timer/counter_reg__0 [3]),
        .I4(\u_100ms_timer/counter_reg__0 [2]),
        .I5(tick_10ms),
        .O(p_0_in__10[0]));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT4 #(
    .INIT(16'hA6AA)) 
    \hb_error_counter[1]_i_1 
       (.I0(\hb_error_counter_reg_n_0_[1] ),
        .I1(tick_10ms),
        .I2(\hb_error_counter[2]_i_2_n_0 ),
        .I3(hb_alert),
        .O(p_0_in__10[1]));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'hA6AAAAAA)) 
    \hb_error_counter[2]_i_1 
       (.I0(hb_error),
        .I1(hb_alert),
        .I2(\hb_error_counter[2]_i_2_n_0 ),
        .I3(tick_10ms),
        .I4(\hb_error_counter_reg_n_0_[1] ),
        .O(p_0_in__10[2]));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT4 #(
    .INIT(16'hFFDF)) 
    \hb_error_counter[2]_i_2 
       (.I0(\u_100ms_timer/counter_reg__0 [0]),
        .I1(\u_100ms_timer/counter_reg__0 [1]),
        .I2(\u_100ms_timer/counter_reg__0 [3]),
        .I3(\u_100ms_timer/counter_reg__0 [2]),
        .O(\hb_error_counter[2]_i_2_n_0 ));
  FDRE \hb_error_counter_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in__10[0]),
        .Q(hb_alert),
        .R(rst));
  FDRE \hb_error_counter_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in__10[1]),
        .Q(\hb_error_counter_reg_n_0_[1] ),
        .R(rst));
  FDRE \hb_error_counter_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in__10[2]),
        .Q(hb_error),
        .R(rst));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT3 #(
    .INIT(8'h9A)) 
    \hb_normal_counter[0]_i_1 
       (.I0(hb_normal_counter),
        .I1(\hb_normal_counter[1]_i_2_n_0 ),
        .I2(tick_10ms),
        .O(p_0_in[0]));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT4 #(
    .INIT(16'hA6AA)) 
    \hb_normal_counter[1]_i_1 
       (.I0(hb_normal),
        .I1(tick_10ms),
        .I2(\hb_normal_counter[1]_i_2_n_0 ),
        .I3(hb_normal_counter),
        .O(p_0_in[1]));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'hFFFFEFFF)) 
    \hb_normal_counter[1]_i_2 
       (.I0(\u_250ms_timer/counter_reg__0 [1]),
        .I1(\u_250ms_timer/counter_reg__0 [0]),
        .I2(\u_250ms_timer/counter_reg__0 [3]),
        .I3(\u_250ms_timer/counter_reg__0 [4]),
        .I4(\u_250ms_timer/counter_reg__0 [2]),
        .O(\hb_normal_counter[1]_i_2_n_0 ));
  FDRE \hb_normal_counter_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[0]),
        .Q(hb_normal_counter),
        .R(rst));
  FDRE \hb_normal_counter_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[1]),
        .Q(hb_normal),
        .R(rst));
  LUT4 #(
    .INIT(16'h22B2)) 
    i__carry_i_1
       (.I0(breath_pwm_reg__0[7]),
        .I1(\u_breath/cnt_reg__0 [7]),
        .I2(breath_pwm_reg__0[6]),
        .I3(\u_breath/cnt_reg__0 [6]),
        .O(i__carry_i_1_n_0));
  LUT4 #(
    .INIT(16'h22B2)) 
    i__carry_i_2
       (.I0(breath_pwm_reg__0[5]),
        .I1(\u_breath/cnt_reg__0 [5]),
        .I2(breath_pwm_reg__0[4]),
        .I3(\u_breath/cnt_reg__0 [4]),
        .O(i__carry_i_2_n_0));
  LUT4 #(
    .INIT(16'h44D4)) 
    i__carry_i_3
       (.I0(\u_breath/cnt_reg__0 [3]),
        .I1(breath_pwm_reg__0[3]),
        .I2(breath_pwm_reg__0[2]),
        .I3(\u_breath/cnt_reg__0 [2]),
        .O(i__carry_i_3_n_0));
  LUT2 #(
    .INIT(4'h2)) 
    i__carry_i_4
       (.I0(breath_pwm_reg__0[1]),
        .I1(\u_breath/cnt_reg__0 [1]),
        .O(i__carry_i_4_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    i__carry_i_5
       (.I0(\u_breath/cnt_reg__0 [7]),
        .I1(breath_pwm_reg__0[7]),
        .I2(\u_breath/cnt_reg__0 [6]),
        .I3(breath_pwm_reg__0[6]),
        .O(i__carry_i_5_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    i__carry_i_6
       (.I0(\u_breath/cnt_reg__0 [5]),
        .I1(breath_pwm_reg__0[5]),
        .I2(\u_breath/cnt_reg__0 [4]),
        .I3(breath_pwm_reg__0[4]),
        .O(i__carry_i_6_n_0));
  LUT4 #(
    .INIT(16'h9009)) 
    i__carry_i_7
       (.I0(breath_pwm_reg__0[3]),
        .I1(\u_breath/cnt_reg__0 [3]),
        .I2(\u_breath/cnt_reg__0 [2]),
        .I3(breath_pwm_reg__0[2]),
        .O(i__carry_i_7_n_0));
  LUT3 #(
    .INIT(8'h41)) 
    i__carry_i_8
       (.I0(\u_breath/cnt_reg__0 [0]),
        .I1(breath_pwm_reg__0[1]),
        .I2(\u_breath/cnt_reg__0 [1]),
        .O(i__carry_i_8_n_0));
  LUT5 #(
    .INIT(32'h00903903)) 
    out_carry_i_1
       (.I0(breath_mode[0]),
        .I1(\u_breath_tick/counter_reg__0 [3]),
        .I2(breath_mode[2]),
        .I3(breath_mode[1]),
        .I4(\u_breath_tick/counter_reg__0 [4]),
        .O(out_carry_i_1_n_0));
  LUT6 #(
    .INIT(64'h0088082018001001)) 
    out_carry_i_2
       (.I0(\u_breath_tick/counter_reg__0 [1]),
        .I1(\u_breath_tick/counter_reg__0 [2]),
        .I2(breath_mode[1]),
        .I3(breath_mode[2]),
        .I4(breath_mode[0]),
        .I5(\u_breath_tick/counter_reg__0 [0]),
        .O(out_carry_i_2_n_0));
  LUT5 #(
    .INIT(32'h00200000)) 
    tick_100ms_INST_0
       (.I0(tick_10ms),
        .I1(\u_100ms_timer/counter_reg__0 [2]),
        .I2(\u_100ms_timer/counter_reg__0 [3]),
        .I3(\u_100ms_timer/counter_reg__0 [1]),
        .I4(\u_100ms_timer/counter_reg__0 [0]),
        .O(tick_100ms));
  LUT6 #(
    .INIT(64'h0000000000000001)) 
    tick_100us_INST_0
       (.I0(tick_10us_INST_0_i_2_n_0),
        .I1(\u_base_timer/counter_reg__0 [3]),
        .I2(\u_base_timer/counter_reg__0 [4]),
        .I3(\u_base_timer/counter_reg__0 [2]),
        .I4(tick_10us_INST_0_i_1_n_0),
        .I5(tick_100us_INST_0_i_1_n_0),
        .O(tick_100us));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT4 #(
    .INIT(16'hFFDF)) 
    tick_100us_INST_0_i_1
       (.I0(\u_100us_timer/counter_reg__0 [0]),
        .I1(\u_100us_timer/counter_reg__0 [1]),
        .I2(\u_100us_timer/counter_reg__0 [3]),
        .I3(\u_100us_timer/counter_reg__0 [2]),
        .O(tick_100us_INST_0_i_1_n_0));
  LUT6 #(
    .INIT(64'h0000000000000010)) 
    tick_10ms_INST_0
       (.I0(tick_1ms_INST_0_i_2_n_0),
        .I1(tick_10us_INST_0_i_2_n_0),
        .I2(tick_1ms_INST_0_i_1_n_0),
        .I3(tick_10us_INST_0_i_1_n_0),
        .I4(tick_100us_INST_0_i_1_n_0),
        .I5(tick_10ms_INST_0_i_1_n_0),
        .O(tick_10ms));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT4 #(
    .INIT(16'hFFDF)) 
    tick_10ms_INST_0_i_1
       (.I0(\u_10ms_timer/counter_reg__0 [0]),
        .I1(\u_10ms_timer/counter_reg__0 [1]),
        .I2(\u_10ms_timer/counter_reg__0 [3]),
        .I3(\u_10ms_timer/counter_reg__0 [2]),
        .O(tick_10ms_INST_0_i_1_n_0));
  LUT5 #(
    .INIT(32'h00000001)) 
    tick_10us_INST_0
       (.I0(tick_10us_INST_0_i_1_n_0),
        .I1(\u_base_timer/counter_reg__0 [2]),
        .I2(\u_base_timer/counter_reg__0 [4]),
        .I3(\u_base_timer/counter_reg__0 [3]),
        .I4(tick_10us_INST_0_i_2_n_0),
        .O(tick_10us));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT4 #(
    .INIT(16'h7FFF)) 
    tick_10us_INST_0_i_1
       (.I0(\u_base_timer/counter_reg__0 [1]),
        .I1(\u_base_timer/counter_reg__0 [0]),
        .I2(\u_base_timer/counter_reg__0 [6]),
        .I3(\u_base_timer/counter_reg__0 [5]),
        .O(tick_10us_INST_0_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT4 #(
    .INIT(16'hFFDF)) 
    tick_10us_INST_0_i_2
       (.I0(\u_10us_timer/counter_reg__0 [0]),
        .I1(\u_10us_timer/counter_reg__0 [1]),
        .I2(\u_10us_timer/counter_reg__0 [3]),
        .I3(\u_10us_timer/counter_reg__0 [2]),
        .O(tick_10us_INST_0_i_2_n_0));
  LUT5 #(
    .INIT(32'h00000010)) 
    tick_1ms_INST_0
       (.I0(tick_100us_INST_0_i_1_n_0),
        .I1(tick_10us_INST_0_i_1_n_0),
        .I2(tick_1ms_INST_0_i_1_n_0),
        .I3(tick_10us_INST_0_i_2_n_0),
        .I4(tick_1ms_INST_0_i_2_n_0),
        .O(tick_1ms));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT3 #(
    .INIT(8'h01)) 
    tick_1ms_INST_0_i_1
       (.I0(\u_base_timer/counter_reg__0 [2]),
        .I1(\u_base_timer/counter_reg__0 [4]),
        .I2(\u_base_timer/counter_reg__0 [3]),
        .O(tick_1ms_INST_0_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT4 #(
    .INIT(16'hFFDF)) 
    tick_1ms_INST_0_i_2
       (.I0(\u_1ms_timer/counter_reg__0 [0]),
        .I1(\u_1ms_timer/counter_reg__0 [1]),
        .I2(\u_1ms_timer/counter_reg__0 [3]),
        .I3(\u_1ms_timer/counter_reg__0 [2]),
        .O(tick_1ms_INST_0_i_2_n_0));
  LUT6 #(
    .INIT(64'h0000000000200000)) 
    tick_1s_INST_0
       (.I0(\u_100ms_timer/counter_reg__0 [0]),
        .I1(\u_100ms_timer/counter_reg__0 [1]),
        .I2(\u_100ms_timer/counter_reg__0 [3]),
        .I3(\u_100ms_timer/counter_reg__0 [2]),
        .I4(tick_10ms),
        .I5(tick_1s_INST_0_i_1_n_0),
        .O(tick_1s));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT4 #(
    .INIT(16'hFFDF)) 
    tick_1s_INST_0_i_1
       (.I0(\u_1s_timer/counter_reg__0 [0]),
        .I1(\u_1s_timer/counter_reg__0 [1]),
        .I2(\u_1s_timer/counter_reg__0 [3]),
        .I3(\u_1s_timer/counter_reg__0 [2]),
        .O(tick_1s_INST_0_i_1_n_0));
  LUT6 #(
    .INIT(64'h0000000001000000)) 
    tick_1us_INST_0
       (.I0(\u_base_timer/counter_reg__0 [3]),
        .I1(\u_base_timer/counter_reg__0 [4]),
        .I2(\u_base_timer/counter_reg__0 [2]),
        .I3(\u_base_timer/counter_reg__0 [5]),
        .I4(\u_base_timer/counter_reg__0 [6]),
        .I5(tick_1us_INST_0_i_1_n_0),
        .O(tick_1us));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT2 #(
    .INIT(4'h7)) 
    tick_1us_INST_0_i_1
       (.I0(\u_base_timer/counter_reg__0 [0]),
        .I1(\u_base_timer/counter_reg__0 [1]),
        .O(tick_1us_INST_0_i_1_n_0));
  LUT6 #(
    .INIT(64'h0000000000002000)) 
    tick_250ms_INST_0
       (.I0(tick_10ms),
        .I1(\u_250ms_timer/counter_reg__0 [2]),
        .I2(\u_250ms_timer/counter_reg__0 [4]),
        .I3(\u_250ms_timer/counter_reg__0 [3]),
        .I4(\u_250ms_timer/counter_reg__0 [0]),
        .I5(\u_250ms_timer/counter_reg__0 [1]),
        .O(tick_250ms));
  FDRE \u_100ms_timer/counter_reg[0] 
       (.C(clk),
        .CE(tick_10ms),
        .D(p_0_in__5[0]),
        .Q(\u_100ms_timer/counter_reg__0 [0]),
        .R(\counter[3]_i_1__3_n_0 ));
  FDRE \u_100ms_timer/counter_reg[1] 
       (.C(clk),
        .CE(tick_10ms),
        .D(p_0_in__5[1]),
        .Q(\u_100ms_timer/counter_reg__0 [1]),
        .R(\counter[3]_i_1__3_n_0 ));
  FDRE \u_100ms_timer/counter_reg[2] 
       (.C(clk),
        .CE(tick_10ms),
        .D(p_0_in__5[2]),
        .Q(\u_100ms_timer/counter_reg__0 [2]),
        .R(\counter[3]_i_1__3_n_0 ));
  FDRE \u_100ms_timer/counter_reg[3] 
       (.C(clk),
        .CE(tick_10ms),
        .D(p_0_in__5[3]),
        .Q(\u_100ms_timer/counter_reg__0 [3]),
        .R(\counter[3]_i_1__3_n_0 ));
  FDRE \u_100us_timer/counter_reg[0] 
       (.C(clk),
        .CE(tick_10us),
        .D(p_0_in__2[0]),
        .Q(\u_100us_timer/counter_reg__0 [0]),
        .R(\counter[3]_i_1__0_n_0 ));
  FDRE \u_100us_timer/counter_reg[1] 
       (.C(clk),
        .CE(tick_10us),
        .D(p_0_in__2[1]),
        .Q(\u_100us_timer/counter_reg__0 [1]),
        .R(\counter[3]_i_1__0_n_0 ));
  FDRE \u_100us_timer/counter_reg[2] 
       (.C(clk),
        .CE(tick_10us),
        .D(p_0_in__2[2]),
        .Q(\u_100us_timer/counter_reg__0 [2]),
        .R(\counter[3]_i_1__0_n_0 ));
  FDRE \u_100us_timer/counter_reg[3] 
       (.C(clk),
        .CE(tick_10us),
        .D(p_0_in__2[3]),
        .Q(\u_100us_timer/counter_reg__0 [3]),
        .R(\counter[3]_i_1__0_n_0 ));
  FDRE \u_10ms_timer/counter_reg[0] 
       (.C(clk),
        .CE(tick_1ms),
        .D(p_0_in__4[0]),
        .Q(\u_10ms_timer/counter_reg__0 [0]),
        .R(\counter[3]_i_1__2_n_0 ));
  FDRE \u_10ms_timer/counter_reg[1] 
       (.C(clk),
        .CE(tick_1ms),
        .D(p_0_in__4[1]),
        .Q(\u_10ms_timer/counter_reg__0 [1]),
        .R(\counter[3]_i_1__2_n_0 ));
  FDRE \u_10ms_timer/counter_reg[2] 
       (.C(clk),
        .CE(tick_1ms),
        .D(p_0_in__4[2]),
        .Q(\u_10ms_timer/counter_reg__0 [2]),
        .R(\counter[3]_i_1__2_n_0 ));
  FDRE \u_10ms_timer/counter_reg[3] 
       (.C(clk),
        .CE(tick_1ms),
        .D(p_0_in__4[3]),
        .Q(\u_10ms_timer/counter_reg__0 [3]),
        .R(\counter[3]_i_1__2_n_0 ));
  FDRE \u_10us_timer/counter_reg[0] 
       (.C(clk),
        .CE(tick_1us),
        .D(p_0_in__1[0]),
        .Q(\u_10us_timer/counter_reg__0 [0]),
        .R(\counter[3]_i_1_n_0 ));
  FDRE \u_10us_timer/counter_reg[1] 
       (.C(clk),
        .CE(tick_1us),
        .D(p_0_in__1[1]),
        .Q(\u_10us_timer/counter_reg__0 [1]),
        .R(\counter[3]_i_1_n_0 ));
  FDRE \u_10us_timer/counter_reg[2] 
       (.C(clk),
        .CE(tick_1us),
        .D(p_0_in__1[2]),
        .Q(\u_10us_timer/counter_reg__0 [2]),
        .R(\counter[3]_i_1_n_0 ));
  FDRE \u_10us_timer/counter_reg[3] 
       (.C(clk),
        .CE(tick_1us),
        .D(p_0_in__1[3]),
        .Q(\u_10us_timer/counter_reg__0 [3]),
        .R(\counter[3]_i_1_n_0 ));
  FDRE \u_1ms_timer/counter_reg[0] 
       (.C(clk),
        .CE(tick_100us),
        .D(p_0_in__3[0]),
        .Q(\u_1ms_timer/counter_reg__0 [0]),
        .R(\counter[3]_i_1__1_n_0 ));
  FDRE \u_1ms_timer/counter_reg[1] 
       (.C(clk),
        .CE(tick_100us),
        .D(p_0_in__3[1]),
        .Q(\u_1ms_timer/counter_reg__0 [1]),
        .R(\counter[3]_i_1__1_n_0 ));
  FDRE \u_1ms_timer/counter_reg[2] 
       (.C(clk),
        .CE(tick_100us),
        .D(p_0_in__3[2]),
        .Q(\u_1ms_timer/counter_reg__0 [2]),
        .R(\counter[3]_i_1__1_n_0 ));
  FDRE \u_1ms_timer/counter_reg[3] 
       (.C(clk),
        .CE(tick_100us),
        .D(p_0_in__3[3]),
        .Q(\u_1ms_timer/counter_reg__0 [3]),
        .R(\counter[3]_i_1__1_n_0 ));
  FDRE \u_1s_timer/counter_reg[0] 
       (.C(clk),
        .CE(tick_100ms),
        .D(p_0_in__7[0]),
        .Q(\u_1s_timer/counter_reg__0 [0]),
        .R(\counter[3]_i_1__4_n_0 ));
  FDRE \u_1s_timer/counter_reg[1] 
       (.C(clk),
        .CE(tick_100ms),
        .D(p_0_in__7[1]),
        .Q(\u_1s_timer/counter_reg__0 [1]),
        .R(\counter[3]_i_1__4_n_0 ));
  FDRE \u_1s_timer/counter_reg[2] 
       (.C(clk),
        .CE(tick_100ms),
        .D(p_0_in__7[2]),
        .Q(\u_1s_timer/counter_reg__0 [2]),
        .R(\counter[3]_i_1__4_n_0 ));
  FDRE \u_1s_timer/counter_reg[3] 
       (.C(clk),
        .CE(tick_100ms),
        .D(p_0_in__7[3]),
        .Q(\u_1s_timer/counter_reg__0 [3]),
        .R(\counter[3]_i_1__4_n_0 ));
  FDRE \u_250ms_timer/counter_reg[0] 
       (.C(clk),
        .CE(tick_10ms),
        .D(p_0_in__6[0]),
        .Q(\u_250ms_timer/counter_reg__0 [0]),
        .R(\counter[4]_i_1_n_0 ));
  FDRE \u_250ms_timer/counter_reg[1] 
       (.C(clk),
        .CE(tick_10ms),
        .D(p_0_in__6[1]),
        .Q(\u_250ms_timer/counter_reg__0 [1]),
        .R(\counter[4]_i_1_n_0 ));
  FDRE \u_250ms_timer/counter_reg[2] 
       (.C(clk),
        .CE(tick_10ms),
        .D(p_0_in__6[2]),
        .Q(\u_250ms_timer/counter_reg__0 [2]),
        .R(\counter[4]_i_1_n_0 ));
  FDRE \u_250ms_timer/counter_reg[3] 
       (.C(clk),
        .CE(tick_10ms),
        .D(p_0_in__6[3]),
        .Q(\u_250ms_timer/counter_reg__0 [3]),
        .R(\counter[4]_i_1_n_0 ));
  FDRE \u_250ms_timer/counter_reg[4] 
       (.C(clk),
        .CE(tick_10ms),
        .D(p_0_in__6[4]),
        .Q(\u_250ms_timer/counter_reg__0 [4]),
        .R(\counter[4]_i_1_n_0 ));
  FDRE \u_base_timer/counter_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in__0[0]),
        .Q(\u_base_timer/counter_reg__0 [0]),
        .R(\counter[6]_i_1_n_0 ));
  FDRE \u_base_timer/counter_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in__0[1]),
        .Q(\u_base_timer/counter_reg__0 [1]),
        .R(\counter[6]_i_1_n_0 ));
  FDRE \u_base_timer/counter_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\counter[2]_i_1__7_n_0 ),
        .Q(\u_base_timer/counter_reg__0 [2]),
        .R(\counter[6]_i_1_n_0 ));
  FDRE \u_base_timer/counter_reg[3] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in__0[3]),
        .Q(\u_base_timer/counter_reg__0 [3]),
        .R(\counter[6]_i_1_n_0 ));
  FDRE \u_base_timer/counter_reg[4] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in__0[4]),
        .Q(\u_base_timer/counter_reg__0 [4]),
        .R(\counter[6]_i_1_n_0 ));
  FDRE \u_base_timer/counter_reg[5] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in__0[5]),
        .Q(\u_base_timer/counter_reg__0 [5]),
        .R(\counter[6]_i_1_n_0 ));
  FDRE \u_base_timer/counter_reg[6] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in__0[6]),
        .Q(\u_base_timer/counter_reg__0 [6]),
        .R(\counter[6]_i_1_n_0 ));
  FDRE \u_breath/cnt_reg[0] 
       (.C(clk),
        .CE(tick_1us),
        .D(\cnt[0]_i_1_n_0 ),
        .Q(\u_breath/cnt_reg__0 [0]),
        .R(rst));
  FDRE \u_breath/cnt_reg[1] 
       (.C(clk),
        .CE(tick_1us),
        .D(p_0_in__9[1]),
        .Q(\u_breath/cnt_reg__0 [1]),
        .R(rst));
  FDRE \u_breath/cnt_reg[2] 
       (.C(clk),
        .CE(tick_1us),
        .D(p_0_in__9[2]),
        .Q(\u_breath/cnt_reg__0 [2]),
        .R(rst));
  FDRE \u_breath/cnt_reg[3] 
       (.C(clk),
        .CE(tick_1us),
        .D(p_0_in__9[3]),
        .Q(\u_breath/cnt_reg__0 [3]),
        .R(rst));
  FDRE \u_breath/cnt_reg[4] 
       (.C(clk),
        .CE(tick_1us),
        .D(p_0_in__9[4]),
        .Q(\u_breath/cnt_reg__0 [4]),
        .R(rst));
  FDRE \u_breath/cnt_reg[5] 
       (.C(clk),
        .CE(tick_1us),
        .D(p_0_in__9[5]),
        .Q(\u_breath/cnt_reg__0 [5]),
        .R(rst));
  FDRE \u_breath/cnt_reg[6] 
       (.C(clk),
        .CE(tick_1us),
        .D(p_0_in__9[6]),
        .Q(\u_breath/cnt_reg__0 [6]),
        .R(rst));
  FDRE \u_breath/cnt_reg[7] 
       (.C(clk),
        .CE(tick_1us),
        .D(p_0_in__9[7]),
        .Q(\u_breath/cnt_reg__0 [7]),
        .R(rst));
  CARRY4 \u_breath/pwm0_inferred__0/i__carry 
       (.CI(1'b0),
        .CO({\u_breath/pwm0_inferred__0/i__carry_n_0 ,\u_breath/pwm0_inferred__0/i__carry_n_1 ,\u_breath/pwm0_inferred__0/i__carry_n_2 ,\u_breath/pwm0_inferred__0/i__carry_n_3 }),
        .CYINIT(1'b1),
        .DI({i__carry_i_1_n_0,i__carry_i_2_n_0,i__carry_i_3_n_0,i__carry_i_4_n_0}),
        .O(\NLW_u_breath/pwm0_inferred__0/i__carry_O_UNCONNECTED [3:0]),
        .S({i__carry_i_5_n_0,i__carry_i_6_n_0,i__carry_i_7_n_0,i__carry_i_8_n_0}));
  FDRE \u_breath_tick/counter_reg[0] 
       (.C(clk),
        .CE(tick_1ms),
        .D(p_0_in__8[0]),
        .Q(\u_breath_tick/counter_reg__0 [0]),
        .R(\counter[4]_i_1__1_n_0 ));
  FDRE \u_breath_tick/counter_reg[1] 
       (.C(clk),
        .CE(tick_1ms),
        .D(p_0_in__8[1]),
        .Q(\u_breath_tick/counter_reg__0 [1]),
        .R(\counter[4]_i_1__1_n_0 ));
  FDRE \u_breath_tick/counter_reg[2] 
       (.C(clk),
        .CE(tick_1ms),
        .D(p_0_in__8[2]),
        .Q(\u_breath_tick/counter_reg__0 [2]),
        .R(\counter[4]_i_1__1_n_0 ));
  FDRE \u_breath_tick/counter_reg[3] 
       (.C(clk),
        .CE(tick_1ms),
        .D(p_0_in__8[3]),
        .Q(\u_breath_tick/counter_reg__0 [3]),
        .R(\counter[4]_i_1__1_n_0 ));
  FDRE \u_breath_tick/counter_reg[4] 
       (.C(clk),
        .CE(tick_1ms),
        .D(p_0_in__8[4]),
        .Q(\u_breath_tick/counter_reg__0 [4]),
        .R(\counter[4]_i_1__1_n_0 ));
  CARRY4 \u_breath_tick/out_carry 
       (.CI(1'b0),
        .CO({\NLW_u_breath_tick/out_carry_CO_UNCONNECTED [3:2],\u_breath_tick/out_carry_n_2 ,\u_breath_tick/out_carry_n_3 }),
        .CYINIT(1'b1),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O(\NLW_u_breath_tick/out_carry_O_UNCONNECTED [3:0]),
        .S({1'b0,1'b0,out_carry_i_1_n_0,out_carry_i_2_n_0}));
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
