import time

import polars as pl

from src.config import Config, load_config
from src.logger import logging
from src.utils import encode_transcripts, extract_transform_load

TRANSCRIPTS_FILE: str = "video_transcripts.parquet"
EMBEDDINGS_FILE: str = "transcript_embeddings.parquet"


def main() -> None:
    start: float = time.time()

    # a list of pl.LazyFrames, one per YouTube channel ID
    lazyframes: list[pl.LazyFrame] = [
        extract_transform_load(channel_id) for channel_id in load_config().youtube_channel_ids
    ]

    # vertically concatenate the list of pl.LazyFrames into a single pl.LazyFrame, ...
    # convert it to a pl.DataFrame, and write it to ../data/video_transcripts.parquet
    (
        pl.concat(lazyframes, how="vertical")
        .sort("datetime")
        .collect()
        .write_parquet(Config.Path.DATA_DIR / TRANSCRIPTS_FILE)
    )

    # read in ../data/video_transcripts.parquet as a pl.DataFrame, generate embeddings for 
    # each YouTube video transcript and write the resulting pl.DataFrame to ...
    # ../data/transcript_embeddings.parquet
    encode_transcripts(TRANSCRIPTS_FILE, EMBEDDINGS_FILE)
    logging.info(
        "Finished! Processing time was ~%s minutes.", f"{((time.time() - start)/60):.2f}"
    )


if __name__ == "__main__":
    main()
