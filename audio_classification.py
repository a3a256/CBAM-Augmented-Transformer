import os
import subprocess

import pandas as pd
import numpy as np

from custom_dataloader import SoundDS

from torch.utils.data import DataLoader

from model_scripts.fine_tuned_pann import load_fine_tuned_pann
from model_scripts.pann_cbam_transformer import load_pann_cbam_transformer
from model_scripts.pann_cbam_encoder import load_pann_cbam_encoder
from model_scripts.pann_transformer_cbam_gate import load_pann_transformer_cbam_gate

import torch.nn.functional as F

import torch
import torch.nn as nn

from model_scripts.ast_model import ASTModel
from ast_dataloader import AudiosetDataset


# """Extracts an audio segment using ffmpeg and runs SSLAM inference."""
# # Create temp segment file
# segment_file = "segment_mpsenetmvdr-mic3_8array-up-File1.wav"
# # Base ffmpeg command
# cmd = "ffmpeg -i mpsenetmvdr-mic3_8array-up-File1.wav -ss 0.000000 -to 30.710669"
# # Add channel selection if specified
# # if channel is not None:
# #     # For 7.1 audio files, we need to use the pan filter to extract specific channels
# #     # Channel mapping for 7.1: 0=FL, 1=FR, 2=FC, 3=LFE, 4=BL, 5=BR, 6=SL, 7=SR
# #     cmd += f" -af 'pan=mono|c0=c{channel}'"
# cmd += " {} -y".format(segment_file)
# subprocess.run(cmd, shell=True, check=True)

human_labels = {
        # Speech & spoken communication
        "Speech", "Male speech, man speaking", "Female speech, woman speaking",
        "Child speech, kid speaking", "Conversation", "Narration, monologue",
        "Babbling", "Speech synthesizer",

        # Distress / high-arousal vocalizations — most SAR-relevant category
        "Shout", "Bellow", "Whoop", "Yell", "Battle cry", "Children shouting",
        "Screaming", "Whispering",

        # Laughter
        "Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle",

        # Crying / distress sounds — critical for your SAR framing
        "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan", "Sigh",

        # Singing (borderline — see note below)
        "Singing", "Choir", "Yodeling", "Chant", "Mantra",
        "Male singing", "Female singing", "Child singing",
        "Rapping", "Humming",

        # Involuntary vocal/breathing sounds
        "Groan", "Grunt", "Whistling", "Breathing", "Wheeze", "Snoring",
        "Gasp", "Pant", "Snort", "Cough", "Throat clearing", "Sneeze", "Sniff",

        # Human non-vocal body actions
        "Run", "Shuffle", "Walk, footsteps", "Chewing, mastication", "Biting",
        "Gargling", "Stomach rumble", "Burping, eructation", "Hiccup", "Fart",

        # Hands / signalling gestures — matches DroneAudioSet's HNV examples
        "Hands", "Finger snapping", "Clapping",

        # Physiological (unlikely in mic recordings, included for completeness)
        "Heart sounds, heartbeat", "Heart murmur",

        # Crowd / group human presence
        "Cheering", "Applause", "Chatter", "Crowd",
        "Hubbub, speech noise, speech babble", "Children playing",
    }


def process_audioset_predictions(top12_labels):
    """Process AudioSet predictions to create a mapping from label index to human/non-human classification."""

    return any(label in human_labels for label in top12_labels)


def get_ground_truth_class(soundclass: str) -> str:
    """Map soundclass to ground truth (H or NH)"""
    human_classes = ['male', 'female', 'crying', 'humansounds']
    return "H" if soundclass in human_classes else "NH"

def process_audioset_labels(as_labels, label_index):
    """Process AudioSet labels to create a mapping from label index to human/non-human classification."""
    return as_labels.iloc[label_index, 2]

def create_df():
    data = {
        "file_path": [],
        "start_time": [],
        "end_time": [],
        "human_nonhuman": [],
        "real_class": []
    }

    for i in range(1, 7):
        with open("ground_truths/File{}.txt".format(i)) as file:
            lines = [line.rstrip() for line in file]

        res = [line.split() for line in lines]

        for line in res:
            data["file_path"].append("mpsenetmvdr-mic3_8array-up-File{}.wav".format(i))
            data["start_time"].append(float(line[0]))
            data["end_time"].append(float(line[1]))
            data["real_class"].append(line[2])
            data["human_nonhuman"].append(get_ground_truth_class(line[2]))
    return pd.DataFrame(data)


def pann_results(df, human_indices, as_labels):
    """Run inference using the PANN model and return predictions."""

    pann_ds = SoundDS(df.values)

    pann_ds.sr = 32000  # Set sample rate to 32kHz for PANN model
    pann_ds.duration = 30000  # Set duration to 32k samples for PANN model

    pann_dl = DataLoader(pann_ds, batch_size=1, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_set_pann = True  # Set to True if the fine-tuned PANN model was trained on AudioSet, False otherwise

    print("Loading PANN model...")
    pann = load_fine_tuned_pann(pre_trained_model_path="pre_trained_models/best_pann_infant_cry.pth", configuration="configurations/best_pann_infant_cry_config.json", audioset=audio_set_pann)
    print("PANN model loaded.")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        pann = nn.DataParallel(pann)

    pann = pann.to(device)

    print("PANN Predictions...")


    pann_preds = []

    pann_confidences = []
    
    
    cur = 1
    with torch.no_grad():
        for data, target in pann_dl:
            data = data.squeeze(0).to(device) 
            if torch.cuda.is_available():
                data = data.to(device)
            # output_ast = ast(data.to(torch.float16))
            output_pann = pann(data)
            score = output_pann[:, human_indices].max(dim=1).values.item()
            pann_confidences.append(score)
            ordered_output = output_pann.argsort(dim=1, descending=True)
            ordered_output = ordered_output.cpu().numpy().tolist()[0][:12]  # Get top 12 predictions
            ordered_output = [process_audioset_labels(as_labels, idx) for idx in ordered_output]
            pann_preds.append(ordered_output)
            print("PANN Processed {}/{} audio segments.".format(cur, len(pann_dl)))
            cur += 1

    print("PANN Predictions completed.")

    predictions = []
    
    human_nonhuman_predictions = []

    for i in pann_preds:
        predictions.append('|'.join(i))
        human_nonhuman_predictions.append(process_audioset_predictions(i))

    human_nonhuman_predictions = ["H" if pred else "NH" for pred in human_nonhuman_predictions]

    pann_df = df.copy()

    pann_df["pann_predict"] = predictions
    pann_df["human_nonhuman_predict"] = human_nonhuman_predictions
    pann_df["confidence"] = pann_confidences

    return pann_df


def ast_results(df, human_indices, as_labels):
    """Run inference using the AST model and return predictions."""

    audio_conf = {
        'num_mel_bins': 128,
        'target_length': 1024,
        'freqm': 0,        # disable SpecAugment entirely for eval
        'timem': 0,
        'mixup': 0,
        'skip_norm': False,
        'mode': 'eval',
        'dataset': 'audioset',
        'noise': False,
        'mean': -4.2677393,   # back to the correct official AST constants
        'std': 4.5689974,
    }

    ast_ds = AudiosetDataset(df.values, audio_conf=audio_conf)
    ast_dl = DataLoader(ast_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading AST model...")
    ast = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=True, model_size='base384')
    print("AST model loaded.")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        ast = nn.DataParallel(ast)

    ast = ast.to(device)

    print("AST Predictions...")


    ast_preds = []
    ast_confidences = []


    cur = 1
    with torch.no_grad():
        for data, target in ast_dl:
            data = data.squeeze(0).to(device) 
            if torch.cuda.is_available():
                data = data.to(device)
            # output_ast = ast(data.to(torch.float16))
            output_ast = ast(data)
            output_ast = F.softmax(output_ast, dim=1)
            avg_probs = output_ast.mean(dim=0, keepdim=True)
            score = avg_probs[:, human_indices].max(dim=1).values.item()
            ast_confidences.append(score)
            ordered_output = avg_probs.argsort(dim=1, descending=True)
            ordered_output = ordered_output.cpu().numpy().tolist()[0][:12]  # Get top 12 predictions
            ordered_output = [process_audioset_labels(as_labels, idx) for idx in ordered_output]
            ast_preds.append(ordered_output)
            print("AST Processed {}/{} audio segments.".format(cur, len(ast_dl)))
            cur += 1

    print("AST Predictions completed.")


    predictions = []

    human_nonhuman_predictions = []

    for i in ast_preds:
        predictions.append('|'.join(i))
        human_nonhuman_predictions.append(process_audioset_predictions(i))


    human_nonhuman_predictions = ["H" if pred else "NH" for pred in human_nonhuman_predictions]

    ast_df = df.copy()

    ast_df["ast_predict"] = predictions
    ast_df["human_nonhuman_predict"] = human_nonhuman_predictions
    ast_df["confidence"] = ast_confidences

    return ast_df

def custom_results(df):
    
    ds = SoundDS(df.values)

    dl = DataLoader(ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    pct = load_pann_cbam_transformer(pre_trained_model_path="pre_trained_models/best_pann_cbam_transformer_infant_cry.pth", configuration="configurations/best_pann_cbam_transformer_infant_cry_config.json")
    pce = load_pann_cbam_encoder(pre_trained_model_path="pre_trained_models/best_pann_cbam_encoder_infant_cry.pth", configuration="configurations/best_pann_cbam_encoder_infant_cry_config.json")
    ptc = load_pann_transformer_cbam_gate(pre_trained_model_path="pre_trained_models/best_pann_transformer_cbam_gate_infant_cry.pth", configuration="configurations/best_pann_transformer_cbam_gate_infant_cry_config.json")

    pct = pct.to(device)
    pce = pce.to(device)
    ptc = ptc.to(device)

    pct = pct.eval()
    pce = pce.eval()
    ptc = ptc.eval()

    print("Models loaded.")

    
    pct_confidences = []
    pce_confidences = []
    ptc_confidences = []
    cur = 1

    with torch.no_grad():
        for data, target in dl:

            if torch.cuda.is_available():
                data = data.cuda()

            
            output_pct = pct(data, False)
            output_pce = pce(data, False)
            output_ptc = ptc(data, False)

            output_pct = F.softmax(output_pct, dim=1)
            output_pce = F.softmax(output_pce, dim=1)
            output_ptc = F.softmax(output_ptc, dim=1)

            p_human = output_pct[:, 1]

            pct_confidences.extend(p_human.cpu().numpy())


            p_human = output_pce[:, 1]
            
            pce_confidences.extend(p_human.cpu().numpy())

            p_human = output_ptc[:, 1]
            
            ptc_confidences.extend(p_human.cpu().numpy())

            print("Processed {}/{} audio segments.".format(cur, len(dl)))
            cur += 1

    custom_df = df.copy()

    custom_df["pct_human_confidences"] = pct_confidences
    custom_df["pce_human_confidences"] = pce_confidences
    custom_df["ptc_human_confidences"] = ptc_confidences

    return custom_df

    



if __name__ == "__main__":

    df = create_df()

    as_labels = pd.read_csv("audioset_class_labels_indices.csv")

    human_indices = [i for i, label in enumerate(as_labels["display_name"]) if label in human_labels]

    ast_df = ast_results(df, human_indices, as_labels)
    pann_df = pann_results(df, human_indices, as_labels)
    custom_df = custom_results(df)

    ast_df.to_csv("ast_predictions.csv", index=False)
    pann_df.to_csv("pann_predictions.csv", index=False)
    custom_df.to_csv("custom_model_predictions.csv", index=False)

