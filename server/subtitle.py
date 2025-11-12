import re
import os 
import uuid
import json

def convert_time_to_srt_format(seconds):
    """Converts seconds to the standard SRT time format (HH:MM:SS,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = round((seconds - int(seconds)) * 1000)

    if milliseconds == 1000:
        milliseconds = 0
        secs += 1
        if secs == 60:
            secs, minutes = 0, minutes + 1
            if minutes == 60:
                minutes, hours = 0, hours + 1

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def word_level_srt(words_timestamp, srt_path="word_level_subtitle.srt", shorts=False):
    """Generates an SRT file with one word per subtitle entry."""
    punctuation = re.compile(r'[.,!?;:"\–—_~^+*|]')
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for i, word_info in enumerate(words_timestamp, start=1):
            start = convert_time_to_srt_format(word_info['start'])
            end = convert_time_to_srt_format(word_info['end'])
            word = re.sub(punctuation, '', word_info['word'])
            if word.strip().lower() == 'i': word = "I"
            if not shorts: word = word.replace("-", "")
            srt_file.write(f"{i}\n{start} --> {end}\n{word}\n\n")



def split_line_by_char_limit(text, max_chars_per_line=38):
    """Splits a string into multiple lines based on a character limit."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line + " " + word) <= max_chars_per_line:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def merge_punctuation_glitches(subtitles):
    """Cleans up punctuation artifacts at the boundaries of subtitle entries."""
    if not subtitles:
        return []

    cleaned = [subtitles[0]]
    for i in range(1, len(subtitles)):
        prev = cleaned[-1]
        curr = subtitles[i]

        prev_text = prev["text"].rstrip()
        curr_text = curr["text"].lstrip()

        match = re.match(r'^([,.:;!?]+)(\s*)(.+)', curr_text)
        if match:
            punct, _, rest = match.groups()
            if not prev_text.endswith(tuple(punct)):
                prev["text"] = prev_text + punct
            curr_text = rest.strip()

        unwanted_chars = ['"', '“', '”', ';', ':']
        for ch in unwanted_chars:
            curr_text = curr_text.replace(ch, '')
        curr_text = curr_text.strip()

        if not curr_text or re.fullmatch(r'[.,!?]+', curr_text):
            prev["end"] = curr["end"]
            continue

        curr["text"] = curr_text
        prev["text"] = prev["text"].replace('"', '').replace('“', '').replace('”', '')
        cleaned.append(curr)

    return cleaned


def write_sentence_srt(
    word_level_timestamps, output_file="subtitles_professional.srt", max_lines=2,
    max_duration_s=7.0, max_chars_per_line=38, hard_pause_threshold=0.5,
    merge_pause_threshold=0.4
):
    """Creates professional-grade SRT files and a corresponding timestamp.json file."""
    if not word_level_timestamps:
        return

    # Phase 1: Generate draft subtitles based on timing and length rules
    draft_subtitles = []
    i = 0
    while i < len(word_level_timestamps):
        start_time = word_level_timestamps[i]["start"]

        # We'll now store the full word objects, not just the text
        current_word_objects = []

        j = i
        while j < len(word_level_timestamps):
            entry = word_level_timestamps[j]

            # Create potential text from the word objects
            potential_words = [w["word"] for w in current_word_objects] + [entry["word"]]
            potential_text = " ".join(potential_words)

            if len(split_line_by_char_limit(potential_text, max_chars_per_line)) > max_lines: break
            if (entry["end"] - start_time) > max_duration_s and current_word_objects: break

            if j > i:
                prev_entry = word_level_timestamps[j-1]
                pause = entry["start"] - prev_entry["end"]
                if pause >= hard_pause_threshold: break
                if prev_entry["word"].endswith(('.','!','?')): break

            # Append the full word object
            current_word_objects.append(entry)
            j += 1

        if not current_word_objects:
            current_word_objects.append(word_level_timestamps[i])
            j = i + 1

        text = " ".join([w["word"] for w in current_word_objects])
        end_time = word_level_timestamps[j - 1]["end"]

        # Include the list of word objects in our draft subtitle
        draft_subtitles.append({
            "start": start_time,
            "end": end_time,
            "text": text,
            "words": current_word_objects
        })
        i = j

    # Phase 2: Post-process to merge single-word "orphan" subtitles
    if not draft_subtitles: return
    final_subtitles = [draft_subtitles[0]]
    for k in range(1, len(draft_subtitles)):
        prev_sub = final_subtitles[-1]
        current_sub = draft_subtitles[k]
        is_orphan = len(current_sub["text"].split()) == 1
        pause_from_prev = current_sub["start"] - prev_sub["end"]

        if is_orphan and pause_from_prev < merge_pause_threshold:
            merged_text = prev_sub["text"] + " " + current_sub["text"]
            if len(split_line_by_char_limit(merged_text, max_chars_per_line)) <= max_lines:
                prev_sub["text"] = merged_text
                prev_sub["end"] = current_sub["end"]

                # Merge the word-level data as well
                prev_sub["words"].extend(current_sub["words"])
                continue

        final_subtitles.append(current_sub)

    final_subtitles = merge_punctuation_glitches(final_subtitles)

    # This dictionary will hold the data for our JSON file
    timestamps_data = {}

    # Phase 3: Write the final SRT file (and prepare JSON data)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, sub in enumerate(final_subtitles, start=1):
            # --- SRT Writing (Unchanged) ---
            text = sub["text"].replace(" ,", ",").replace(" .", ".")
            formatted_lines = split_line_by_char_limit(text, max_chars_per_line)
            start_time_str = convert_time_to_srt_format(sub['start'])
            end_time_str = convert_time_to_srt_format(sub['end'])

            f.write(f"{idx}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            f.write("\n".join(formatted_lines) + "\n\n")

            # Create the list of word dictionaries for the current subtitle
            word_data = []
            for word_obj in sub["words"]:
                word_data.append({
                    "word": word_obj["word"],
                    "start": convert_time_to_srt_format(word_obj["start"]),
                    "end": convert_time_to_srt_format(word_obj["end"])
                })

            # Add the complete entry to our main dictionary
            timestamps_data[str(idx)] = {
                "text": "\n".join(formatted_lines),
                "start": start_time_str,
                "end": end_time_str,
                "words": word_data
            }

    # Write the collected data to the JSON file
    json_output_file = output_file.replace(".srt",".json")
    with open(json_output_file, "w", encoding="utf-8") as f_json:
        json.dump(timestamps_data, f_json, indent=4, ensure_ascii=False)

    # print(f"Successfully generated SRT file: {output_file}")
    # print(f"Successfully generated JSON file: {json_output_file}")
    return json_output_file
def make_subtitle(word_level_timestamps,file_path):
  os.makedirs("./subtitles/",exist_ok=True)
  file_name = os.path.splitext(os.path.basename(file_path))[0]
  unique_id = str(uuid.uuid4())[:6] 
  word_level_srt_file=f"./subtitles/{file_name}_subtitle_word_level_{unique_id}.srt"
  sentence_srt_file=f"./subtitles/{file_name}_subtitle_sentences_{unique_id}.srt"
  shorts_srt_file=f"./subtitles/{file_name}_subtitle_reels_{unique_id}.srt"
  word_level_srt(
      word_level_timestamps,
      srt_path=word_level_srt_file,
      shorts=False
  )

  sentence_json = write_sentence_srt(
      word_level_timestamps,
      output_file=sentence_srt_file,
      max_lines=2,
      max_duration_s=7.0,
      max_chars_per_line=38,
      hard_pause_threshold=0.5,
      merge_pause_threshold=0.4
  )

  shorts_json = write_sentence_srt(
      word_level_timestamps,
      output_file=shorts_srt_file,
      max_lines=1,
      max_duration_s=2.0,
      max_chars_per_line=17
  )
  return sentence_srt_file,word_level_srt_file,shorts_srt_file
