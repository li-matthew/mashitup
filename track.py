#delta of feature from start
#running average of features

from bs4 import BeautifulSoup
import os
import glob
import essentia.standard as es
from datetime import datetime
import librosa
import numpy as np
import scipy
import pandas as pd

tracklists = glob.glob("/Users/mashen/mashitup_mixes/*.html")
mixes = glob.glob("/Users/mashen/mashitup_mixes/*.mp3")

def get_timestamps():
    for tracklist in tracklists:
        print(tracklist)
        with open(tracklist, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'lxml')

        tracks = []
        t = []
        # print(soup)

        track_rows = soup.find_all('div', id=lambda x: x and x.startswith('tlp_'))
        for row in track_rows:
            time_element = row.find('div', class_='cue')
            timestamp = time_element.text.strip() if time_element else "00:00"
            tracks.append(timestamp)

            colons = timestamp.count(':')
            if colons == 1:
                dt = datetime.strptime(timestamp, "%M:%S")
                t.append(dt.minute * 60 + dt.second)
            elif colons == 2:
                dt = datetime.strptime(timestamp, "%H:%M:%S")
                t.append(dt.hour * 3600 + dt.minute * 60 + dt.second)
        print(tracks)
        t.append(int(librosa.get_duration(path=mixes[0])))
        if t[0] != 0:
            t.insert(0, 0)
        print(t)
        return t

def ess(times):
    result = []
    for mix in mixes:

        for time in range(len(times) -1 ):
            segment, sr = librosa.load(mix, sr=44100, offset=times[time], duration=times[time+1] - times[time])
            
            # y_harm, y_perc = librosa.effects.hpss(segment)
            key_data = es.KeyExtractor()(segment)

            rhythm_data = es.RhythmExtractor2013()(segment)

            loudness = es.Loudness()(segment)
            rms = es.RMS()(segment)
            energy = es.Energy()(segment)
            danceability, d_curve = es.Danceability()(segment)
            dynamics = es.DynamicComplexity()(segment)
            onset_rate = es.OnsetRate()(segment)
            
            

            spectrum = es.Spectrum()(es.Windowing()(segment))
            centroid = es.Centroid()(spectrum)
            flatness = es.Flatness()(spectrum)
            hfc = es.HFC()(spectrum)
            entropy = es.Entropy()(spectrum)
            crest = es.Crest()(spectrum)

            zcr = es.ZeroCrossingRate()(segment)
            flux = es.Flux()(segment)

            data = {
                "mix": mix,
                "track": time,
                "key": key_data,
                "bpm": rhythm_data[0],
                "loudness": loudness,
                "rms": rms,
                "energy": energy,
                "danceability": danceability,
                "onset_rate": onset_rate,
                "dynamics": dynamics,
                "entropy": entropy,
                "crest": crest,
                "centroid": centroid,
                "flatness": flatness,
                "hfc": hfc,
                "zcr": zcr,
                "flux": flux
            }

            result.append(data)

    df = pd.DataFrame(result)
    print(df)
ess(get_timestamps())
