import copy
import random
import shlex
import os
import datetime
from rembg import remove, new_session
from PIL import Image
import io

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers, errors
from modules.processing import Processed, process_images
from modules.shared import state

suffix_lists = {
    'Standard Emotions (28 imgs)': [
        'admiration=admiration, loving expression, admiring',
        'amusement=amused, laughing, smiling, bright eyes, relaxed, open mouth',
        'anger=angry, upset, frown',
        'annoyance=annoyed, half-closed eyes, frown',
        'approval=approval, approving smile, happy',
        'caring=loving smile, gentle smile, caring expression, gentle expression',
        'confusion=confused',
        'curiosity=interested expression, curious, bright eyes, :o',
        'desire=longing expression, desire, horny',
        'disappointment=disappointed, frowning, unhappy',
        'disapproval=disapproving expression, empty eyes',
        'disgust=hate, disgust, disgusted expression, green blush',
        'embarrassment=shy, embarrassed, averting gaze, blushing',
        'excitement=happy, excited, open mouth, smile',
        'fear=fear, afraid, panic, teary eyes, narrow pupils',
        'gratitude=loving eyes, grateful expression, thanking, loving expression',
        'grief=grief, sad, tears, crying',
        'joy=laughing, smile, happy, very happy, warm smile',
        'love=ahegao, aroused, loving, love, heart eyes, heart pupils, smile',
        'nervousness=nervous, blushing, averted gaze, surprised, nervous eyes',
        'neutral=neutral expression, mild smile',
        'optimism=smile, optimistic expression, half closed happy eyes',
        'pride=smug, prideful, half open eyes, happy, open mouth smile',
        'realization=eyes wide open, :o, surprise',
        'relief=relief, ahegao, relaxed, cumming, orgasm face',
        'remorse=sad, remorseful, empty eyes',
        'sadness=sad, very sad, tears',
        'surprise=open mouth, :o, surprised',
    ],
}

def process_string_tag(tag):
    return tag

def process_int_tag(tag):
    return int(tag)

def process_float_tag(tag):
    return float(tag)

def process_boolean_tag(tag):
    return True if (tag == "true") else False

prompt_tags = {
    "sd_model": None,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag
}

def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        assert pos+1 < len(args), f'missing argument for command line option {arg}'

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt":
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue

        func = prompt_tags.get(tag, None)
        assert func, f'unknown commandline option: {arg}'

        val = args[pos+1]
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)

        res[tag] = func(val)

        pos += 2

    return res

class Script(scripts.Script):

    def title(self):
        return "SillyTavern: inpaint expressions"

    def ui(self, is_img2img):
        suffix_checkboxes = []
        suffix_textboxes = []

        # Create checkbox and textbox for each suffix
        for suffix in suffix_lists['Standard Emotions (28 imgs)']:
            name, suff = suffix.split('=')
            checkbox = gr.Checkbox(value=True, label=name, elem_id=name)
            textbox = gr.Textbox(value=suff, label=f"{name} suffix", elem_id=f"{name}_suffix")
            suffix_checkboxes.append(checkbox)
            suffix_textboxes.append(textbox)

        # Add buttons to check/uncheck all checkboxes
        check_all_button = gr.Button("Check All")
        uncheck_all_button = gr.Button("Uncheck All")

        def check_all():
            return [True] * len(suffix_checkboxes)

        def uncheck_all():
            return [False] * len(suffix_checkboxes)

        check_all_button.click(check_all, [], suffix_checkboxes)
        uncheck_all_button.click(uncheck_all, [], suffix_checkboxes)

        return [*suffix_checkboxes, *suffix_textboxes, check_all_button, uncheck_all_button]

    def run(self, p, *args):
        suffix_checkboxes_values = args[:28]
        suffix_textboxes_values = args[28:]

        suffixes = suffix_lists['Standard Emotions (28 imgs)']
        suffixes = [suffix for suffix, selected in zip(suffixes, suffix_checkboxes_values) if selected]

        p.do_not_save_grid = True

        job_count = 0
        jobs = []

        for suffix, custom_suffix in zip(suffixes, suffix_textboxes_values):
            name, _ = suffix.split('=')
            new_prompt = p.prompt + " " + custom_suffix
            args = {"prompt": new_prompt, "name": name}

            job_count += args.get("n_iter", p.n_iter)
            jobs.append(args)

        print(f"Will process {len(suffixes)} suffixes in {job_count} jobs.")
        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.job_count = job_count

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_folder = f"outputs/expressions_packs/expressions_{current_time}"
        os.makedirs(output_folder, exist_ok=True)

        images = []
        all_prompts = []
        infotexts = []
        for args in jobs:
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                setattr(copy_p, k, v)

            proc = process_images(copy_p)
            current_images = []

            for image in proc.images:
                current_images.append(image)
                images.append(image)

            for img, name in zip(current_images, [args['name']] * len(current_images)):
                img.save(os.path.join(output_folder, f"{name}.png"))

            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        return Processed(p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)
