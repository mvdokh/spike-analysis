Data Needed:

spikes.csv 
    this contains "time", unit, and electrode
    "time" must be multiplied by 30,000 to match pico intervals

pico.csv
    this contains interval start, interval end, interval duration
    (generated through whisker toolbox)
    !!!: There is an absolute start interval and end interval (0 frame duration) 

electrode.cfg
    this contains spatial layout of electrode
    columns: electrode, electrode, ML coord, DV coord

Pre Processing Required:
1. multiply 1st column spikes.csv by 30,000
2. create new file for each interval duration 
