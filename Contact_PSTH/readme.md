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

usage:
python Contact_PSTH/contact_psth.py --data_dir "C:\Users\wanglab\Desktop\Club Like Endings\102625_1"
python Contact_PSTH/contact_psth_combined.py --data_dir "C:\Users\wanglab\Desktop\Club Like Endings\102625_1"
python Contact_PSTH/contact_psth_to_csv.py --data_dir "C:\Users\wanglab\Desktop\Club Like Endings\102725_1"
python Contact_PSTH/profile_units.py --data_dir "C:\Users\wanglab\Desktop\Club Like Endings\102725_1"
python Contact_PSTH/tuning_curves.py --data_dir "C:\Users\wanglab\Desktop\Club Like Endings\102225_2"
python "Contact_PSTH\unit_analysis_suite.py" --data_dir "C:\Users\wanglab\Desktop\Club Like Endings\102225_2"
python Contact_PSTH\contact_psth_end_aligned.py --data_dir "C:\Users\wanglab\Desktop\Club Like Endings\102225_2"
python Contact_PSTH\population_psth_heatmap.py --data_dir "C:\Users\wanglab\Desktop\Club Like Endings\102225_1"

"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\interval_0_mask_contact_retraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\interval_1_mask_contact.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\interval_1_mask_contact_protraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\interval_1_mask_contact_retraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\interval_2_mask_contact.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\interval_2_mask_contact_protraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\interval_2_mask_contact_retraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\interval_3_mask_contact.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\interval_3_mask_contact_protraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\interval_3_mask_contact_retraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\contact_intervals.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\interval_0_mask_contact.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102525_1\per_whisker_contact\interval_0_mask_contact_protraction.csv"


"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\contact_intervals.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_0_shrunk.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_0_shrunk_protraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_0_shrunk_retraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_1_shrunk.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_1_shrunk_protraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_1_shrunk_retraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_2_shrunk.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_2_shrunk_protraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_2_shrunk_retraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_3_shrunk.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_3_shrunk_protraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_3_shrunk_retraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_4_shrunk.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_4_shrunk_protraction.csv"
"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\per_whisker_contact\interval_4_shrunk_retraction.csv"

dsi_raw = (Rret - Rpro) / (Rret + Rpro) - standard dsi
dsi_adj = DSIraw x (2 x sqrt (nret x npro) / nret + npro) - adjusted by trial count balance 
balance = (2sqrt(nret*npro) / nret + npro) - 1.0 = equal trials -> 0 = extreme imbalance

wsi_raw = (Rthis - Rothers) / (Rthis + Rothers) - how much this whiskers peak FR exceeds the meanof the other whiskers
wsi_adj = WSIraw x nthis/nmax - scaled by the ratio of this whiskers trial count to the most sampled whiskers trial count


Tuning Curve Significance Testing:
Kruskal-Wallis test on bin-level firing rates, followed by pairwise Mann-Whitney U tests with multiple comparison correction. 

For retraction versus protraction within a whisker, a Mann-Whitney U test would work similarly

Winsorize the modulation indices to clip extreme values (units with near-zero baseline get 33x indices)
Use robust scaling (IQR-based) instead of z-score so outliers don't dominate
Use Calinski-Harabasz as a secondary criterion — silhouette loves splitting off tiny outlier clusters

too many features