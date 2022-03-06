////////////////////////////////////////////////////////////////////////////////
// Stream fifo using internally cascaded SRLC32E blocks in a CLB
////////////////////////////////////////////////////////////////////////////////

`default_nettype none

module stream_fifo_srl32 #(
    parameter DATA_WIDTH = 32,

    //max allowed capacity between 1 and 128
    parameter FIFO_DEPTH = 32
) (
    input wire clk,
    input wire rst,

    input wire [DATA_WIDTH-1:0] in_tdata,
    input wire in_tlast,
    input wire in_tvalid,
    output reg in_tready,

    output wire [DATA_WIDTH-1:0] out_tdata,
    output wire out_tlast,
    output reg out_tvalid,
    input wire out_tready
);
    localparam FULL_WIDTH = DATA_WIDTH+1; //DATA + LAST

    //how many SRL32s to cascade?
    localparam DEPTH = (FIFO_DEPTH+31)/32;

    //complete input and output array for each instantiation
    wire [FULL_WIDTH-1:0] D = {in_tlast, in_tdata};
    wire [FULL_WIDTH-1:0] O;
    assign {out_tlast, out_tdata} = O;

    //logic for read and write cycles
    wire write = in_tvalid && in_tready;
    wire read = out_tvalid && out_tready;

    //increment the address when read and write operations occur
    localparam R_W = $clog2(FIFO_DEPTH);
    reg [R_W-1:0] raddr = {R_W{1'b1}};
    always @(posedge clk) begin
        if (rst) begin
            raddr <= {R_W{1'b1}};
            in_tready <= 1'b1;
            out_tvalid <= 1'b0;
        end
        else if (!read && write) begin
            if (raddr == FIFO_DEPTH-2) in_tready <= 1'b0;
            raddr <= raddr + 1'b1;
            out_tvalid <= 1'b1;
        end
        else if (read && !write) begin
            if (raddr == 0) out_tvalid <= 1'b0;
            raddr <= raddr - 1'b1;
            in_tready <= 1'b1;
        end
    end

    //generate cascaded SRLC32E for each bit
    wire [7:0] raddr7 = {{(8-R_W){1'b0}}, raddr};
    genvar i, j;
    generate
    for (i = 0; i < FULL_WIDTH ; i = i + 1) begin: gen_bits

        wire [3:0] Q;
        wire [4:0] Q31;

        //feed data into the first Q31
        //the rest are between SRL blocks
        assign Q31[0] = D[i];

        //assign output based on upper addr bits
        assign O[i] = Q[raddr7[6:5]];

        for (j = 0; j < DEPTH; j = j + 1) begin: gen_depth
            SRLC32E srlc32e_i_j(
                .Q(Q[j]),
                .Q31(Q31[j+1]),
                .A(raddr7[4:0]),
                .CE(write),
                .CLK(clk),
                .D(Q31[j])
            );
        end

    end
    endgenerate

endmodule //stream_fifo_srl32

`default_nettype wire
