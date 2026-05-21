import argparse

from src.config import Config
from src.pipeline_factory import build_default_pipeline


def main():
    parser = argparse.ArgumentParser(description='画像差分検出ツール')
    parser.add_argument('--origin', type=str, default='imgs/origin', help='元データディレクトリ')
    parser.add_argument('--dummy', type=str, default='imgs/dummy', help='比較対象ディレクトリ')
    parser.add_argument('--threshold', type=float, help='マッチング閾値')
    parser.add_argument('--output', type=str, default='output', help='出力ディレクトリ')
    parser.add_argument('--method', type=str, default='diff', choices=['diff', 'matchshapes'], help='類似度計算方法')
    parser.add_argument('--canny1', type=float, default=50.0, help='Canny閾値1')
    parser.add_argument('--canny2', type=float, default=150.0, help='Canny閾値2')
    parser.add_argument('--no-photo-normalize', action='store_true', help='実写のCAD風正規化を無効化')

    args = parser.parse_args()

    config_kwargs = {
        'output_dir': args.output,
        'canny_threshold1': args.canny1,
        'canny_threshold2': args.canny2,
    }
    if args.threshold is not None:
        config_kwargs['match_threshold'] = args.threshold
    if args.no_photo_normalize:
        config_kwargs['photo_normalize_enabled'] = False
    config = Config(**config_kwargs)

    pipeline = build_default_pipeline(config)

    print(f"Loading images from {args.origin} and {args.dummy}...")
    pairs = pipeline.load_directory(args.origin, args.dummy)
    print(f"Found {len(pairs)} image pairs")

    results = []
    for origin_file, dummy_file, origin, dummy in pairs:
        print(f"Processing: {dummy_file}")

        result = pipeline.process_pair(origin, dummy, method=args.method)
        result.origin_path = f"{args.origin}/{origin_file}"
        result.dummy_path = f"{args.dummy}/{dummy_file}"

        output_name = dummy_file.replace('.', '_')
        image_path = f"{args.output}/{output_name}_result.png"
        json_path = f"{args.output}/{output_name}_result.json"

        pipeline.export_result(result, image_path, json_path)
        results.append(result)

        print(f"  Similarity: {result.similarity_score:.3f}")
        print(f"  Photo normalized: {result.photo_normalized}")
        print(f"  Match: {result.is_match}")

    match_count = sum(1 for r in results if r.is_match)
    print(f"\nSummary: {match_count}/{len(results)} matches")


if __name__ == '__main__':
    main()
