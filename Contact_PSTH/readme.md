"C:\Users\wanglab\Desktop\Club Like Endings\102225_1\per_whisker_contact\interval_0_mask_contact.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102225_1\per_whisker_contact\interval_1_mask_contact.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102225_1\per_whisker_contact\interval_2_mask_contact.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102225_1\per_whisker_contact\interval_3_mask_contact.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102225_1\per_whisker_contact\interval_4_mask_contact.csv"
Start,End
23094,23096
23745,23753


"C:\Users\wanglab\Desktop\Club Like Endings\102225_1\spikes.csv"
     0.11110,   22,   18
     0.49227,   22,   18
     0.68773,   12,   27

"C:\Users\wanglab\Desktop\Club Like Endings\102225_1\time.dat"
"C:\Users\wanglab\Desktop\Club Like Endings\102225_1\digitalin.dat"

[
    {
        "filepath": "102225_1.mp4",
        "data_type": "video",
        "name": "media"
    },
   {
        "filepath": "digitalin.dat",
        "data_type": "time",
        "name": "time",
        "format": "uint16",
        "channel": 1,
        "transition": "rising"
    },
    {
        "filepath": "digitalin.dat",
        "data_type": "time",
        "name": "master",
        "format": "uint16_length"
    },
    {
        "filepath": "amplifier.dat",
        "data_type": "analog",
        "name": "voltage",
        "format": "int16",
        "header_size": 0,
        "channel_count": 32,
        "clock": "master"
    },
    {
        "filepath": "digitalin.dat",
        "data_type": "digital_interval",
        "name": "pico",
        "format": "uint16",
        "channel": 0,
        "transition": "rising",
        "clock": "master",
        "header_size": 0
    },
    {
        "filepath": "spikes.csv",
        "data_type": "digital_event",
        "name": "spikes",
        "format": "csv",
        "scale": 30000,
        "scale_divide": false,
        "identifier_column": 1,
        "clock": "master"

    }
]


C:\Users\wanglab\Desktop\Tongue-Whisker-Analysis\Spike PSTH Pipeline\3_PSTH\digitalin_loading\digitalin_loader.py
C:\Users\wanglab\Desktop\Tongue-Whisker-Analysis\Spike PSTH Pipeline\3_PSTH\digitalin_loading\psth_digitalin_analysis.py

Video frames → synced via TTL rising edges on channel 1 of digitalin.dat (each rising edge = one frame)
Spike times → in seconds, referenced to the master clock (30kHz sample rate)
Frame N maps to the Nth rising edge sample index, converted to seconds via sample_index / 30000

'''
python Contact_PSTH/contact_psth.py --data_dir "C:\Users\wanglab\Desktop\Club Like Endings\102625_1"
'''