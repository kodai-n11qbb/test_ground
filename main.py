import argparse

from src.config import Config
from src.pipeline_factory import build_default_pipeline


def main():
    parser = argparse.ArgumentParser(description='画像差分検出ツール')
    parser.add_argument('--origin', type=str, default='imgs/origin', help='元データディレクトリ')
    parser.add_argument('--dummy', type=str, default='imgs/dummy', help='比較対象ディレクトリ')
    parser.add_argument('--threshold', type=float, help='マッチング閾値')
    parser.add_argument('--output', type=str, default='output', help='出力ディレクトリ')
    parser.add_argument('--method', type=str, default='iou', choices=['diff', 'matchshapes', 'iou'], help='類似度計算方法')
    parser.add_argument('--canny1', type=float, default=30.0, help='Canny閾値1')
    parser.add_argument('--canny2', type=float, default=100.0, help='Canny閾値2')
    parser.add_argument('--no-photo-normalize', action='store_true', help='実写のCAD風正規化を無効化')

    args = parser.parse_args()

    # まず config.json から設定をロードする
    config = Config.load_from_json()

    # CLI 引数で指定されたものがあれば上書きする
    config.output_dir = args.output
    config.origin_dir = args.origin
    config.dummy_dir = args.dummy
    config.canny_threshold1 = args.canny1
    config.canny_threshold2 = args.canny2
    if args.threshold is not None:
        config.match_threshold = args.threshold
    if args.no_photo_normalize:
        config.photo_normalize_enabled = False
    if args.method:
        config.match_method = args.method

    pipeline = build_default_pipeline(config)

    print(f"Loading images from {config.origin_dir} and {config.dummy_dir}...")
    pairs = pipeline.load_directory(config.origin_dir, config.dummy_dir)
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
