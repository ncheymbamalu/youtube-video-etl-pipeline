import json
import os

import numpy as np
import polars as pl
import requests

from dotenv import load_dotenv
from requests import Response
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi

from src.config import Config
from src.logger import logging

load_dotenv()


# NOTE: ./.venv/lib/python3.10/site-packages/pytube/extract.py needs to be modified, so ...
# that the 'pytube' library's 'Channel' class can properly read the YouTube video URLs ...
# that are listed in ./config.yaml
# source: https://stackoverflow.com/questions/74957606/pytube-and-the-new-channel-urls


def extract_transform_load(youtube_channel_id: str, max_results: int = 50) -> pl.LazyFrame:
    """
    Extracts the ID, creation datetime, title, and transcription of several
    YouTube videos, and returns a pl.LazyFrame containing the extracted data.
    NOTE: the following link, https://www.youtube.com/watch?v=qPKmPaNaCmE&t=1s,
    shows how to find any YouTube channel ID

    Args:
        youtube_channel_id (str): Unique ID for the YouTube channel of interest
        max_results (int): Maximum number of transcribed YouTube videos that will
        be extracted. Defaults to 50, which is the maximum number of results allowed
        by the YouTube API.
    """
    try:
        logging.info(
            "Extracting and transcribing video data from the YouTube channel ID, '%s'.",
            youtube_channel_id
        )
        params: dict[str, int | list[str] | str] = {
            "key": os.getenv("YOUTUBE_API_KEY"),
            "channelId": youtube_channel_id,
            "part": ["snippet", "id"],
            "order": "date",
            "maxResults": max_results
        }
        response: Response = requests.get(
            "https://www.googleapis.com/youtube/v3/search", params=params
        )
        video_records: list[dict[str, str]] = []
        for item in json.loads(response.text).get("items"):
            try:
                video_record: dict[str, str] = {
                    "video_id": item.get("id").get("videoId"),
                    "datetime": item.get("snippet").get("publishedAt"),
                    "title": item.get("snippet").get("title"),
                    "transcript": " ".join(
                        transcript_dict.get("text") for transcript_dict in
                        YouTubeTranscriptApi.get_transcript(item.get("id").get("videoId"))
                    )
                }
                video_records.append(video_record)
            except:
                logging.info(
                    "Video ID, '%s', doesn't have a transcript.", item.get("id").get("videoId")
                )
        return (
            pl.LazyFrame(video_records)
            .with_columns(
                pl.col("datetime").cast(pl.Datetime),
                pl.col("title").str.replace_many(["&#39;", "&amp;", "  "], ["'", "&", " "]),
                pl.col("transcript").str.replace_many(["&#39;", "&amp;", "  "], ["'", "&", " "])
            )
            .unique(subset="video_id")
            .sort(by="datetime")
        )
    except Exception as e:
        raise e


def encode_transcripts(
        input_file: str, output_file: str, model_name: str = "thenlper/gte-base"
) -> None:
    """Converts YouTube video transcripts (text) to embeddings (vectors)

    Args:
        input_file (str): Name of the file that contains the video transcripts
        output_file (str): Name of the file that the embeddings will be written to
        model_name (str): Name of the embedding model. Defaults to 'thenlper/gte-base'.
    """
    try:
        # instantiate the embedding model and extract its embedding dimension
        model: SentenceTransformer = SentenceTransformer(model_name)
        dmodel: int = model.get_sentence_embedding_dimension()

        # read in the data containing the YouTube video transcripts as a pl.DataFrame
        data: pl.DataFrame = pl.read_parquet(Config.Path.DATA_DIR / input_file)

        # convert each transcript to a dmodel-length vector of embeddings
        # NOTE: the 'embeddings' np.ndarray has shape (N, dmodel), that is, ...
        # N embedding vectors, one per transcript, where each has length dmodel
        embeddings: np.ndarray = model.encode(data["transcript"].to_list())
        schema: dict[str, type] = dict(zip(
            [f"embedding_{idx + 1}" for idx in range(dmodel)], [float] * dmodel
        ))

        # horizontally concatenate the 'data' pl.DataFrame with a pl.DataFrame that ...
        # contains the embeddings, and then write to ../data/ as a parquet
        (
            pl.concat((data, pl.DataFrame(embeddings, schema=schema)), how="horizontal")
            .write_parquet(Config.Path.DATA_DIR / output_file)
        )
        logging.info(
            "Transcript embeddings have been generated and written to %s.",
            Config.Path.DATA_DIR / output_file
        )
    except Exception as e:
        raise e
