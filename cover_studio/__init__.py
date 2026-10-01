"""ピアノカバーを作り直しながら聴き比べる UI (Cover Studio)。

    python -m cover_studio                       # http://127.0.0.1:8100 を開く (プロジェクトは ./projects)
    python -m cover_studio --root D:/covers --port 8080 --no-browser

1 曲 = 1 プロジェクト。音源を入れると tsumugi で原曲の MIDI を作り (重いので 1 回だけ)、その MIDI から設定を変えて
何度でもカバー (テイク) を生成する。テイクごとに設定・seed・モデルを残すので、気に入ったものの設定を戻したり、
途中までをそのまま使って続きだけを作り直したりできる。採譜済みの MIDI を音源と一緒に入れれば採譜は飛ばす。

project.py  プロジェクトとテイクの保存
engine.py   カバーモデルを読んだまま生成する・tsumugi を別プロセスで走らせる
jobs.py     採譜と生成を 1 本の列で順に流す (GPU を取り合わない)
app.py      API (FastAPI) と画面 (web/ のビルド) の配信
"""
