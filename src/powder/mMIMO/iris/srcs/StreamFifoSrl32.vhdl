------------------------------------------------------------------------
-- StreamFifoSrl32
--
-- Stream fifo using internally cascaded SRLC32E blocks in a CLB
------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

library work;

entity StreamFifoSrl32 is
    generic(
        FIFO_DEPTH      : positive := 32 --1 to 128
    );
    port(
        clk : in std_logic;
        rst : in std_logic;

        -- input bus
        in_data : in std_logic_vector;
        in_last : in std_logic := '0';
        in_valid : in std_logic;
        in_ready : out std_logic;

        -- output bus
        out_data : out std_logic_vector;
        out_last : out std_logic;
        out_valid : out std_logic;
        out_ready : in std_logic
    );
end entity StreamFifoSrl32;

architecture rtl of StreamFifoSrl32 is

    component stream_fifo_srl32
    generic (
        DATA_WIDTH      : positive := 32;
        FIFO_DEPTH      : positive := 32
    );
    port (
        clk : in std_logic;
        rst : in std_logic;
        in_tdata : in std_logic_vector(DATA_WIDTH-1 downto 0);
        in_tlast : in std_logic;
        in_tvalid : in std_logic;
        in_tready : out std_logic;
        out_tdata : out std_logic_vector(DATA_WIDTH-1 downto 0);
        out_tlast : out std_logic;
        out_tvalid : out std_logic;
        out_tready : in std_logic
    );
    end component;
begin

    fifo: stream_fifo_srl32
    generic map (
        DATA_WIDTH => in_data'length,
        FIFO_DEPTH => FIFO_DEPTH
    )
    port map (
        clk => clk,
        rst => rst,

        in_tdata => in_data,
        in_tlast => in_last,
        in_tvalid => in_valid,
        in_tready => in_ready,

        out_tdata => out_data,
        out_tlast => out_last,
        out_tvalid => out_valid,
        out_tready => out_ready
    );
end architecture rtl;
