# PHY Building Blocks

Lower-level Clause 16 DSP, coding, scrambling, header, STS, and normative table modules used internally by [generate_chips][uwb_hrp.generate_chips].

## Forward Error Correction (`uwb_hrp.fec`)

::: uwb_hrp.fec
    options:
      members:
        - rs_encode_block
        - rs_encode
        - rs_encoded_length_bits
        - conv_encode_k3
        - conv_encode_k7

## Synchronization Header (`uwb_hrp.shr`)

::: uwb_hrp.shr

## PHY Header (`uwb_hrp.phr`)

::: uwb_hrp.phr

## Scrambled Timestamp Sequence (`uwb_hrp.sts`)

::: uwb_hrp.sts

## PRBS Scrambler (`uwb_hrp.scrambler`)

::: uwb_hrp.scrambler

## Modulation (`uwb_hrp.modulation`)

::: uwb_hrp.modulation

## Normative Tables (`uwb_hrp.tables`)

::: uwb_hrp.tables
    options:
      members:
        - ChannelSpec
        - PreambleCodeSpec
        - PreambleTimingSpec
        - BPMBPSKParams
        - HPRFModParams
