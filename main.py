import argparse
from src.config import Config
from src.image_loader import ImageLoader
from src.shape_matcher import ShapeMatcher
from src.result_exporter import ResultExporter
from src.models import MatchResult


def main():
    parser = argparse.ArgumentParser(description='画像差分検出ツール')
    parser.add_argument('--origin', type=str, default='imgs/origin', help='元データディレクトリ')
    parser.add_argument('--dummy', type=str, default='imgs/dummy', help='比較対象ディレクトリ')
    parser.add_argument('--threshold', type=float, default=0.73, help='マッチング閾値')
    parser.add_argument('--output', type=str, default='output', help='出力ディレクトリ')
    parser.add_argument('--method', type=str, default='diff', choices=['diff', 'matchshapes'], help='類似度計算方法')
    parser.add_argument('--canny1', type=float, default=50.0, help='Canny閾値1')
    parser.add_argument('--canny2', type=float, default=150.0, help='Canny閾値2')
    
    args = parser.parse_args()
    
    # 設定
    config = Config(
        match_threshold=args.threshold,
        output_dir=args.output,
        canny_threshold1=args.canny1,
        canny_threshold2=args.canny2
    )
    
    # コンポーネント初期化（Dependency Injection）
    loader = ImageLoader()
    matcher = ShapeMatcher(config)
    exporter = ResultExporter()
    
    # 画像読み込み
    print(f"Loading images from {args.origin} and {args.dummy}...")
    pairs = loader.load_directory(args.origin, args.dummy)
    print(f"Found {len(pairs)} image pairs")
    
    # 各ペアを処理
    results = []
    for name, origin, dummy in pairs:
        print(f"Processing: {name}")
        
        # パスを設定
        result = matcher.match_shapes(origin, dummy, method=args.method)
        result.origin_path = f"{args.origin}/{name}"
        result.dummy_path = f"{args.dummy}/{name}"
        
        # 結果出力
        output_name = name.replace('.', '_')
        image_path = f"{args.output}/{output_name}_result.png"
        json_path = f"{args.output}/{output_name}_result.json"
        
        exporter.export_image(result, image_path)
        exporter.export_json(result, json_path)
        
        results.append(result)
        
        print(f"  Similarity: {result.similarity_score:.3f}")
        print(f"  Match: {result.is_match}")
    
    # サマリー
    match_count = sum(1 for r in results if r.is_match)
    print(f"\nSummary: {match_count}/{len(results)} matches")


if __name__ == '__main__':
    main()
