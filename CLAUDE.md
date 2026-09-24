## 環境管理ルール
- このプロジェクトは `uv` を使用して Python 環境と依存関係を管理しています。
- Python バージョン: 3.12
- 仮想環境: `.venv` (uv によって管理)

## 実行コマンド
- 依存関係のインストール: `uv pip install -r requirements.txt`
- パッケージの追加: `uv add <package_name>`
- スクリプトの実行: `uv run python <file_name>.py`
- テストの実行: `uv run pytest`

## 開発のガイドライン
- スクリプトを実行する際は、必ず `uv run` を使用してください。これにより正しい仮想環境が自動的に適用されます。
- `pip` や `python` コマンドを直接使用せず、常に `uv` を介して操作してください。
- 新しい依存関係を追加した場合は、`pyproject.toml` または `requirements.txt` が更新されていることを確認してください。