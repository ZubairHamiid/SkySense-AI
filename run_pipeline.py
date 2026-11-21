import argparse
import sys
import logging

from src.pipeline.backfill import run_backfill
from src.pipeline.train import train_and_evaluate_pipeline
from src.pipeline.ingest import run_hourly_ingestion
from src.agents.graph import run_agent_pipeline
from config.settings import DEFAULT_CITY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PipelineRunner")


def main():
    parser = argparse.ArgumentParser(description="AgentWeather Unified Pipeline Runner")
    parser.add_argument(
        "--mode",
        choices=["backfill", "train", "ingest", "agents", "all"],
        default="all",
        help="Pipeline execution mode (default: all)"
    )
    parser.add_argument("--days", type=int, default=180, help="Days of historical data to backfill (default: 180)")
    parser.add_argument("--city", type=str, default=DEFAULT_CITY, help="Target city")
    args = parser.parse_args()

    logger.info(f"Executing AgentWeather Pipeline in [{args.mode.upper()}] mode for {args.city}...")

    if args.mode in ["backfill", "all"]:
        logger.info(f"\n>>> STEP 1: HISTORICAL DATA BACKFILL ({args.days} days) <<<")
        run_backfill(days=args.days, city=args.city)

    if args.mode in ["train", "all"]:
        logger.info("\n>>> STEP 2: MODEL TRAINING & BENCHMARKING <<<")
        train_and_evaluate_pipeline()

    if args.mode == "ingest":
        logger.info("\n>>> STEP 3: HOURLY INGESTION & AGENT CYCLE <<<")
        run_hourly_ingestion(city=args.city, trigger_agents=True)

    if args.mode in ["agents", "all"]:
        logger.info("\n>>> STEP 3: LANGGRAPH AGENTIC REASONING LAYER <<<")
        run_agent_pipeline(city=args.city)

    logger.info("\n=======================================================")
    logger.info("   AGENTWEATHER PIPELINE EXECUTION COMPLETED! 🎉       ")
    logger.info("=======================================================")


if __name__ == "__main__":
    main()

