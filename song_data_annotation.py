import json
import os
import urllib.parse
from dataclasses import dataclass
from typing import Dict, List, Optional

import streamlit as st

# ==========================================
# 定数・設定
# ==========================================
DATA_FILE_PATH = "data/metadata/dataset.json"
PAGE_TITLE = "YouTube Song Mapper"


# ==========================================
# データ管理クラス (Model / Data Access)
# ==========================================
class SongDataManager:
    """データの読み書きと検索ロジックを担当するクラス"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._data: Dict = self._load_data()

    def _load_data(self) -> Dict:
        """JSONファイルを読み込む。存在しない場合は空の辞書を返す。"""
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.error("データファイルの読み込みに失敗しました。")
            return {}

    def save_data(self) -> None:
        """現在のデータをJSONファイルに保存する。"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def get_all_songs(self) -> Dict:
        return self._data

    def get_song(self, song_key: str) -> Optional[Dict]:
        return self._data.get(song_key)

    def update_song(self, key: str, original_id: str, piano_ids: List[str]) -> None:
        """曲データの追加または更新を行う。"""
        self._data[key] = {"original": original_id, "pianos": piano_ids}
        self.save_data()

    def search_songs(self, query: str) -> Dict:
        """曲名(Key)や動画IDで検索を行う。"""
        if not query:
            return self._data

        query = query.lower()
        result = {}

        for key, value in self._data.items():
            # キー（曲名）での検索
            if query in key.lower():
                result[key] = value
                continue

            # IDでの検索 (Original)
            if query in value.get("original", "").lower():
                result[key] = value
                continue

            # IDでの検索 (Pianos)
            if any(query in pid.lower() for pid in value.get("pianos", [])):
                result[key] = value
                continue

        return result

    def check_duplicate_id(self, video_id: str, current_key: str = None) -> List[str]:
        """
        指定されたIDが既に他の曲で使用されていないかチェックする。
        戻り値: 重複している曲キーのリスト
        """
        duplicates = []
        for key, value in self._data.items():
            if key == current_key:
                continue

            # Originalと比較
            if value.get("original") == video_id:
                duplicates.append(f"{key} (Original)")

            # Piano listと比較
            if video_id in value.get("pianos", []):
                duplicates.append(f"{key} (Piano Cover)")

        return duplicates


import re


def extract_youtube_ids(text: str) -> List[str]:
    """
    テキスト内からYouTubeの動画IDを全て抽出する。
    URL(v=...)、短縮URL(youtu.be/...)、またはID単体のいずれにも対応。
    カンマや改行で区切られた入力にも対応。
    """
    if not text:
        return []

    # カンマや空白(改行含む)で分割してトークン化
    tokens = re.split(r"[\s,]+", text)
    results = []

    for token in tokens:
        if not token:
            continue

        # 1. URL形式から抽出 (v=ID, youtu.be/ID, embed/ID)
        # 前後の余計な文字を除去した上でIDパターンを探す
        match = re.search(r"(?:v=|\/|be\/|embed\/|^)([0-9A-Za-z_-]{11})(?:[?&]|$|#)", token)
        if match:
            # group(1)がID
            # ただし、regexの '^' マッチで誤検知する可能性があるため
            # tokenそのものがID形式(11文字)に近いか、あるいはURLの一部かを判断

            candidate = match.group(1)
            # 念のため候補が純粋なID文字種のみか確認
            if re.fullmatch(r"[0-9A-Za-z_-]{11}", candidate):
                results.append(candidate)
                continue

        # 2. そのままIDっぽい文字列 (11桁)
        # URLの一部としてヒットしなかった場合でも、生のIDとしてチェック
        if re.fullmatch(r"[0-9A-Za-z_-]{11}", token):
            results.append(token)

    # 重複排除してリストで返す
    return list(dict.fromkeys(results))


# ==========================================
# UIコンポーネント関数 (View)
# ==========================================
def get_youtube_thumbnail(video_id: str) -> str:
    """YouTubeの動画IDからサムネイルURLを生成する"""
    return f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"


def render_video_preview(label: str, video_id: str):
    """IDが入力されている場合、動画プレイヤーを表示する"""
    if video_id:
        st.caption(f"{label} プレビュー")
        st.video(f"https://www.youtube.com/watch?v={video_id}")
    else:
        st.info(f"{label}のIDを入力するとプレビューが表示されます。")


def render_sidebar_statistics(manager: SongDataManager):
    """サイドバーにデータの統計情報を表示する"""
    data = manager.get_all_songs()

    # --- 集計ロジック ---
    total_songs = len(data)
    total_covers = sum(len(v.get("pianos", [])) for v in data.values())

    # カバー数の多い曲ランキングを作成
    # (曲名, カバー数) のリストにし、カバー数で降順ソート
    ranking = sorted([(k, len(v.get("pianos", []))) for k, v in data.items()], key=lambda x: x[1], reverse=True)

    # Original IDが未登録の曲をカウント
    missing_original = sum(1 for v in data.values() if not v.get("original"))

    # --- UI表示 ---
    st.markdown("### 📊 データ統計")

    # 基本メトリクス (2カラムでコンパクトに)
    c1, c2 = st.columns(2)
    c1.metric("登録楽曲", f"{total_songs} 曲")
    c2.metric("総ペア数", f"{total_covers} 件")

    st.divider()

    # カバー数トップ3
    st.caption("🏆 カバー数 Top 3")
    if ranking:
        for i, (name, count) in enumerate(ranking[:3], 1):
            st.markdown(f"**{i}. {name}** : `{count}` 件")
    else:
        st.text("データがありません")

    # データ不備のアラート（Original IDがないものがあれば表示）
    if missing_original > 0:
        st.divider()
        st.warning(f"⚠️ Original未設定: {missing_original} 曲")
        with st.expander("未設定リストを確認"):
            for k, v in data.items():
                if not v.get("original"):
                    st.text(f"・{k}")


def render_song_card(key: str, data: Dict, on_click_edit):
    """一覧画面での各曲のカード表示"""
    original_id = data.get("original", "")
    piano_count = len(data.get("pianos", []))

    with st.container(border=True):
        col1, col2 = st.columns([1, 2])

        with col1:
            if original_id:
                st.image(get_youtube_thumbnail(original_id), use_container_width=True)
            else:
                st.text("No Image")

        with col2:
            st.subheader(key)
            st.text(f"Original ID: {original_id}")
            st.metric("Piano Covers", f"{piano_count} 件")

            if st.button("編集 / 詳細", key=f"btn_{key}"):
                on_click_edit(key)


# ==========================================
# ページロジック (Controller)
# ==========================================
ITEMS_PER_PAGE = 10  # 1回に読み込む件数


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    # データの初期化
    if "data_manager" not in st.session_state:
        st.session_state.data_manager = SongDataManager(DATA_FILE_PATH)

    manager = st.session_state.data_manager

    # 画面状態の管理
    if "current_view" not in st.session_state:
        st.session_state.current_view = "list"
    if "editing_key" not in st.session_state:
        st.session_state.editing_key = None

    # --- ページネーション状態の管理 ---
    if "display_limit" not in st.session_state:
        st.session_state.display_limit = ITEMS_PER_PAGE
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    # サイドバー：検索と新規作成
    # サイドバー：検索と新規作成
    with st.sidebar:
        st.header("メニュー")
        search_query = st.text_input("検索 (曲名 / YouTube ID)", "")

        # 検索ワードが変わったら表示件数をリセット
        if search_query != st.session_state.last_query:
            st.session_state.display_limit = ITEMS_PER_PAGE
            st.session_state.last_query = search_query

        # === ここに統計情報の表示を追加 ===
        st.divider()
        render_sidebar_statistics(manager)
        # ==============================

        st.divider()

        if st.button("＋ 新規楽曲を追加", type="primary"):
            st.session_state.current_view = "editor"
            st.session_state.editing_key = None
            st.rerun()

        if st.button("一覧に戻る"):
            st.session_state.current_view = "list"
            st.session_state.editing_key = None
            st.session_state.display_limit = ITEMS_PER_PAGE
            st.rerun()

    # ==========================================
    # 画面分岐: 一覧表示モード (遅延読み込み対応)
    # ==========================================
    if st.session_state.current_view == "list":
        st.header("楽曲一覧")

        # 検索実行
        all_matches = manager.search_songs(search_query)
        match_keys = list(all_matches.keys())[::-1]
        total_hits = len(match_keys)

        if total_hits == 0:
            st.warning("該当する楽曲が見つかりません。")
        else:
            # 現在の制限件数までスライスして取得
            current_keys = match_keys[: st.session_state.display_limit]

            # グリッド表示
            cols = st.columns(2)
            for i, key in enumerate(current_keys):
                with cols[i % 2]:

                    def go_to_edit(k=key):
                        st.session_state.current_view = "editor"
                        st.session_state.editing_key = k
                        st.rerun()

                    render_song_card(key, all_matches[key], go_to_edit)

            # 「もっと見る」ボタンの表示判定
            if total_hits > st.session_state.display_limit:
                # 視認性を良くするためのスペース
                st.write("")
                col_center = st.columns([1, 2, 1])
                with col_center[1]:  # 中央寄せ
                    remaining = total_hits - st.session_state.display_limit
                    if st.button(f"もっと見る (残り {remaining} 件)", use_container_width=True):
                        st.session_state.display_limit += ITEMS_PER_PAGE
                        st.rerun()

            # 件数情報の表示（フッター的扱い）
            st.caption(f"表示中: {len(current_keys)} / 全 {total_hits} 件")

    # ==========================================
    # 画面分岐: 編集・詳細モード
    # ==========================================
    elif st.session_state.current_view == "editor":
        is_new = st.session_state.editing_key is None
        current_key = st.session_state.editing_key

        st.header("楽曲データの編集" if not is_new else "新規楽曲の登録")

        current_data = manager.get_song(current_key) if current_key else {"original": "", "pianos": []}

        col_form, col_preview = st.columns([1, 1])

        with col_form:
            # 1. 曲キー（ID）
            input_key = st.text_input(
                "ID / 曲名 (例: AKB48_恋するフォーチュンクッキー)",
                value=current_key if current_key else "",
                disabled=not is_new,
            )

            # === 工夫1: YouTube検索ショートカット ===
            if input_key:
                # 検索クエリ用にキーワードを整形（アンダースコアをスペースに置換など）
                query_word = input_key.replace("_", " ") + " piano"
                encoded_query = urllib.parse.quote(query_word)
                search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
                st.markdown(f"🔎 [YouTubeで '{query_word}' を検索する]({search_url})", unsafe_allow_html=True)
            # ========================================

            st.divider()

            # 2. Original ID (URL貼り付け対応)
            input_original_raw = st.text_input(
                "Original YouTube URL または ID",
                value=current_data.get("original", ""),
                help="ブラウザのURLをそのまま貼り付けてOKです",
            )
            # 入力内容からIDだけを即座に抽出（プレビュー用）
            extracted_originals = extract_youtube_ids(input_original_raw)
            input_original = extracted_originals[0] if extracted_originals else ""

            # 重複チェック (Original)
            if input_original:
                dupes = manager.check_duplicate_id(input_original, current_key)
                if dupes:
                    st.error(f"警告: 既に使用済み: {', '.join(dupes)}")

            # 3. Piano IDs (URL貼り付け対応)
            st.subheader("Piano Covers")
            st.caption("URLまたはIDを貼り付けてください（改行区切りで複数可）")

            current_pianos_str = "\n".join(current_data.get("pianos", []))
            input_pianos_text = st.text_area("Piano YouTube URLs / IDs", value=current_pianos_str, height=200)

            # === 工夫2: URLからのID自動抽出 ===
            # テキストエリアの内容を一度に解析してIDリスト化
            input_pianos_list = extract_youtube_ids(input_pianos_text)

            # 抽出できたIDの数を表示してユーザーにフィードバック
            if input_pianos_list:
                st.info(f"💡 {len(input_pianos_list)} 件のIDを認識しました")
            # ================================

            # 重複チェック (Pianos)
            for pid in input_pianos_list:
                dupes = manager.check_duplicate_id(pid, current_key)
                if dupes:
                    st.warning(f"ID '{pid}' は他で使用されています: {', '.join(dupes)}")

            # 保存ボタン
            if st.button("保存する", type="primary", use_container_width=True):
                if not input_key:
                    st.error("IDを入力してください。")
                else:
                    # 抽出済みのきれいなIDリストを保存
                    manager.update_song(input_key, input_original, input_pianos_list)
                    st.success(f"{input_key} を保存しました！")
                    st.session_state.editing_key = input_key
                    st.rerun()

        # --- プレビュー表示 ---
        with col_preview:
            st.subheader("プレビュー")

            render_video_preview("Original", input_original)

            st.divider()

            if input_pianos_list:
                st.write(f"Piano Covers ({len(input_pianos_list)})")
                st.write(f"Piano Covers ({len(input_pianos_list)})")

                # グリッド表示 (2列)
                p_cols = st.columns(2)
                for i, pid in enumerate(input_pianos_list):
                    with p_cols[i % 2]:
                        st.caption(f"#{i + 1}: {pid}")
                        render_video_preview(f"", pid)
            else:
                st.info("Piano URLを入力するとここにプレビューが表示されます。")


if __name__ == "__main__":
    main()
