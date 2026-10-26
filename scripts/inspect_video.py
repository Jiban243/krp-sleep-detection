import argparse

from src.video_reader import inspect_video, sample_video


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a video for the KRP sleep-detection project."
    )
    parser.add_argument("video", help="Path to an input video")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=5.0,
        help="Approximate number of frames to sample per second (default: 5)",
    )
    args = parser.parse_args()

    info = inspect_video(args.video)

    print("\nVideo information")
    print("-----------------")
    print(f"Path:       {info.path}")
    print(f"Resolution: {info.width} x {info.height}")
    print(f"FPS:        {info.fps:.3f}")
    print(f"Frames:     {info.frame_count}")
    print(f"Duration:   {info.duration_seconds:.2f} seconds")

    sampled_count = sum(
        1 for _ in sample_video(args.video, sample_fps=args.sample_fps)
    )

    print(f"\nSampling rate requested: {args.sample_fps:g} FPS")
    print(f"Frames sampled:           {sampled_count}")


if __name__ == "__main__":
    main()
