# delta of feature from start
# running average of features
# librosa features

from bs4 import BeautifulSoup
import os
import glob
import essentia.standard as es
from datetime import datetime
import librosa
import numpy as np
import scipy
import pandas as pd

CAMELOT_MAP = {
    # Minor Keys (A)
    "Ab minor": "1A",
    "G# minor": "1A",
    "Eb minor": "2A",
    "D# minor": "2A",
    "Bb minor": "3A",
    "A# minor": "3A",
    "F minor": "4A",
    "C minor": "5A",
    "G minor": "6A",
    "D minor": "7A",
    "A minor": "8A",
    "E minor": "9A",
    "B minor": "10A",
    "F# minor": "11A",
    "Gb minor": "11A",
    "C# minor": "12A",
    "Db minor": "12A",
    # Major Keys (B)
    "B major": "1B",
    "F# major": "2B",
    "Gb major": "2B",
    "Db major": "3B",
    "C# major": "3B",
    "Ab major": "4B",
    "G# major": "4B",
    "Eb major": "5B",
    "D# major": "5B",
    "Bb major": "6B",
    "A# major": "6B",
    "F major": "7B",
    "C major": "8B",
    "G major": "9B",
    "D major": "10B",
    "A major": "11B",
    "E major": "12B",
}

CACHE = "/Users/mashen/mashitup_mixes/cache.csv"
cache_exists = os.path.isfile(CACHE)
file_exists = os.path.isfile("/Users/mashen/mashitup_mixes/data.csv")
# tracklists = glob.glob("/Users/mashen/mashitup_mixes/*.html")
mixes = glob.glob("/Users/mashen/mashitup_mixes/*.mp3")


def get_timestamps(tracklist):
    print(tracklist)
    with open(tracklist, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")
    tracks = []
    t = []

    track_rows = soup.find_all("div", id=lambda x: x and x.startswith("tlp_"))
    for row in track_rows:
        time_element = row.find("div", class_="cue")
        timestamp = time_element.text.strip() if time_element else "00:00"
        tracks.append(timestamp)

        colons = timestamp.count(":")
        if colons == 1:
            dt = datetime.strptime(timestamp, "%M:%S")
            t.append(dt.minute * 60 + dt.second)
        elif colons == 2:
            dt = datetime.strptime(timestamp, "%H:%M:%S")
            t.append(dt.hour * 3600 + dt.minute * 60 + dt.second)
    # t.append(int(librosa.get_duration(path=mixes[])))
    if t[0] != 0:
        t.insert(0, 0)
    t = list(set(t))
    t.sort()
    print(t)
    return t


def ess(mix, times):
    result = []
    times.append(int(librosa.get_duration(path=mix)))
    print(times)
    # for mix in mixes:

    for time in range(len(times) - 1):
        segment, sr = librosa.load(
            mix, sr=44100, offset=times[time], duration=times[time + 1] - times[time]
        )
        # downsampled = librosa.resample(segment, orig_sr=sr, target_sr=22050)

        # y_harm, y_perc = librosa.effects.hpss(segment)
        key_data = es.KeyExtractor()(segment)
        key = key_data[0] + " " + key_data[1]
        key = CAMELOT_MAP[key]
        print(key)

        rhythm_data = es.RhythmExtractor2013()(segment)
        bpm = rhythm_data[0]
        if bpm < 100:
            bpm = bpm * 2

        loudness = es.Loudness()(segment)
        rms = es.RMS()(segment)
        energy = es.Energy()(segment)
        danceability, d_curve = es.Danceability()(segment)
        dynamics = es.DynamicComplexity()(segment)
        onset_rate = es.OnsetRate()(segment)
        onset_rate = onset_rate[1]

        if len(segment) % 2 != 0:
            segment = segment[:-1]
        spectrum = es.Spectrum()(es.Windowing()(segment))

        centroid = es.Centroid()(spectrum)
        flatness = es.Flatness()(spectrum + 1e-10)
        hfc = es.HFC()(spectrum)
        entropy = es.Entropy()(spectrum)
        crest = es.Crest()(spectrum)

        zcr = es.ZeroCrossingRate()(segment)
        flux = es.Flux()(segment)

        # LIBROSA FEATURES

        data = {
            "mix": mix.split("/")[-1].split(".")[0],
            "track": time,
            "key": key,
            "bpm": bpm,
            "loudness": loudness,
            "rms": rms,
            "energy": energy,
            "danceability": danceability,
            "onset_rate": onset_rate,
            "dynamic_complexity": dynamics[0],
            "dynamic_loudness": dynamics[1],
            "entropy": entropy,
            "crest": crest,
            "centroid": centroid,
            "flatness": flatness,
            "hfc": hfc,
            "zcr": zcr,
            "flux": flux,
        }

        result.append(data)

    # df = pd.DataFrame(result)
    return result


analyzed = []
if cache_exists:
    analyzed = pd.read_csv(CACHE)
    analyzed = analyzed["0"].to_list()
    print(analyzed)
results = []
names = []

for mix in mixes:
    if mix not in analyzed:
        # print(analyzed)
        tracklist = mix.split(".")[0] + ".html"
        names.append(mix)
        results = results + ess(mix, get_timestamps(tracklist))

pd.DataFrame(names).to_csv(
    "/Users/mashen/mashitup_mixes/cache.csv",
    mode="a",
    index=False,
    header=not file_exists,
    encoding="utf-8-sig",
)
pd.DataFrame(results).to_csv(
    "/Users/mashen/mashitup_mixes/data.csv",
    mode="a",
    index=False,
    header=not file_exists,
    encoding="utf-8-sig",
)

# ess(get_timestamps())
