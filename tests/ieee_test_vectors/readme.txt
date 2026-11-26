DCN:     IEEE P802.15-20-0002-01-004z
Project: IEEE P802.15 Working Group for
         Wireless Personal Area Networks (WPANs)
Title:   15.4z HRP UWB PHY Test Vectors
Date:    20 April, 2020
Source:  Manuel Lafer (NXP Semiconductors)
         Frank Leong (NXP Semiconductors)

Notice:  This document has been prepared to assist the IEEE P802.15.
         It is offered as a basis for discussion and is not binding on
         the contributing individual(s) or organization(s). The
         material in this document is subject to change in form and
         content after further study. The contributor(s) reserve(s) the
         right to add, amend or withdraw material contained herein.
Release: The contributor(s) acknowledge(s) and accept(s) that this
         contribution becomes the property of IEEE and may be made
         publicly available by P802.15.


This readme refers to documents listed below.
The listed abbreviations are used in the descriptions.

Abbreviation  | Document
------------------------
802.15.4_2015 | IEEE Standard for Low-Rate Wireless Personal Area
              | Networks (WPANs) IEEE Std 802.15.4 - 2015
P802.15.4zD08 | P802.15.4z/D08 Draft Standard for
              | Low-Rate Wireless Networks


===============================================
== Description of the "packet_info.txt file" ==
===============================================

The parameter set name is used as title and the according test vector
file in brackets, e.g.:
    BPRF Set #1 (bprf_1.mat)
    ========================

The associated .mat test vector features the BPSK modulation pattern
of the complete packet, stored in "pulses_all".

Variations of the mandatory parameter sets are indicated by an
additional character after the set number, e.g., "BPRF Set #1a"
indicates an optional variation of mandatory BPRF Set #1 from
P802.15.4zD08 Table 57.


The first section "Packet configuration summary" describes the packet
properties. All properties and their description are listed in below.

SHR: Describes which SYNC code index from 802.15.4_2015 Table 16-7
     (802_15_4_code9 - 802_15_4_code24) or P802.15.4zD08 Table 42
     (802_15_4z_code25 - 802_15_4z_code32) is used.
     PSR denotes the number of preamble symbol repetitions.
SFD: Describes the selected SFD pattern as defined by
     phyHrpUwbSfdSelector in P802.15.4zD08 Table 44.
PHR: Describes whether a PHR is present ("PHR @ defined bit rate") or 
     not ("No PHR", e.g., for ranging-only packets). 
     Valid PHR bit rates are as specified in P802.15.4zD08 Table 45.
PSDU Length: Defines the used PSDU length in bytes.
PSDU Data: Defines the used PSDU contents. Packets which are
           20 bytes long contain 0xBABE as repeating pattern,
           whereas the {127,1023}-byte long packets contain 0xAB as
           repeating pattern.
PSDU Modulation: Describes the used modulation type (BPPM_BPSK or
                 BPSK), mean PRF in MHz and symbol rate in MHz.
PSDU Encoding: Describes the used encoding schemes and the 
               resulting bit rate.
STS1 (pre-PSDU): Describes the properties of the STS before the PSDU, 
                 the value "no" indicates that no STS1 is present. 
STS2 (post-PSDU): Describes the properties of the STS after the PSDU, 
                  the value "no" indicates that no STS2 is present.
STS2 Gap: In case a post-PSDU STS is used, this field describes the gap 
          between PSDU and STS if there is any. 
If an STS is present, the described properties are the segment length 
in units of 512 chips as defined in P802.15.4zD08 Table 49 and the 
number of segments as defined in P802.15.4zD08 Table 50,
e.g., "yes, 1 segment, segment length 64".
If an STS is present, the DRBG seed value is also included.


The second section "Packet field index ranges in test vector" describes
which sample indices can be used to access a field of a packet in the
test vector.
The indices are presented in the notation [StartIndex..EndIndex],
indexing starting with index 1. Example: "SYNC: [1..32515]" indicates
that the SYNC field in the test vector starts at index 1 and ends
at index 32515.
A value of "N/A" indicates that this packet does not contain the
specified field.
E.g., "STS1: N/A" indicates that the packet does not contain a
pre-PSDU STS.


=====================================================
== Description of the "plot_4z_hrp_packet.py" file ==
=====================================================

This file is a Python script capable of plotting the test vector files
described in the "packet_info.txt" file.

By default, the script plots test vector file "bprf_1.mat", assuming
the test vector is located in the script's run directory.

To plot a different test vector file, the file name can be specified
via commmand-line parameter. For example, the following can be entered
in the Python prompt to select and plot test vector file "hprf_1.mat".

    python plot_4z_hrp_packet.py -i hprf_1.mat

The built-in help can be accessed as follows.

    python plot_4z_hrp_packet.py -h

The required dependencies are listed in the script's header comments.

