"""
签名验证 REST API 服务

启动方式:
    cd ZZU
    pip install flask
    python api/app.py

接口:
    POST /verify
        Body: multipart/form-data
            sig1: 签名文件1 (.txt)
            sig2: 签名文件2 (.txt)
            threshold: 判决阈值（必填，从 evaluate_svc2004_protocol.py 的 summary.json 取 eer_threshold）
        Response:
            {
                "score": 0.9999,
                "threshold": <用户传入值>,
                "result": "genuine",
                "result_cn": "真签名"
            }

    GET /health
        Response: {"status": "ok"}
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

from data.feature_extractor import load_signature_txt, extract_temporal_features

MAX_LEN = 400
DEFAULT_CHECKPOINT = "outputs/checkpoints/best_model.h5"

MODEL_CONFIG = {
    'input_size': 23,
    'hidden_size': 256,
    'num_layers': 2,
    'rnn_type': 'lstm',
    'bidirectional': True,
    'dropout': 0.2,
    'use_attention': True,
    'attention_hidden_size': 128,
    'distance_metric': 'concat',
    'output_activation': 'sigmoid',
}

app = Flask(__name__)
_model = None


def _fix_length(features: np.ndarray) -> np.ndarray:
    n = features.shape[0]
    if n >= MAX_LEN:
        return features[:MAX_LEN]
    return np.vstack([features, np.zeros((MAX_LEN - n, features.shape[1]), dtype=np.float32)])


def _load_model():
    global _model
    if _model is not None:
        return _model
    from models.siamese import build_siamese_network
    model = build_siamese_network(MODEL_CONFIG)
    dummy = tf.zeros((1, MAX_LEN, 23))
    model((dummy, dummy), training=False)
    checkpoint = os.environ.get('CHECKPOINT_PATH', DEFAULT_CHECKPOINT)
    model.load_weights(checkpoint)
    _model = model
    print(f"模型加载完成: {checkpoint}")
    return _model


def _extract(filepath: str) -> np.ndarray:
    raw = load_signature_txt(filepath)
    if raw.shape[0] == 0:
        raise ValueError(f"无法读取签名文件: {filepath}")
    x, y, t, p = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    return _fix_length(extract_temporal_features(x, y, p, time=t))


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/verify', methods=['POST'])
def verify():
    # 检查文件是否上传
    if 'sig1' not in request.files or 'sig2' not in request.files:
        return jsonify({"error": "需要上传 sig1 和 sig2 两个签名文件"}), 400

    if 'threshold' not in request.form:
        return jsonify({"error": "缺少必填参数 threshold（从 evaluate_svc2004_protocol.py 的 summary.json 取 eer_threshold）"}), 400
    threshold = float(request.form['threshold'])

    # 保存临时文件
    tmp_files = []
    try:
        feats = []
        for key in ['sig1', 'sig2']:
            f = request.files[key]
            if not f.filename.endswith('.TXT') and not f.filename.endswith('.txt'):
                return jsonify({"error": f"{key} 必须是 .txt 文件"}), 400
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
            f.save(tmp.name)
            tmp_files.append(tmp.name)
            feats.append(_extract(tmp.name))

        model = _load_model()
        inp1 = tf.expand_dims(feats[0], 0)
        inp2 = tf.expand_dims(feats[1], 0)
        score = float(model((inp1, inp2), training=False).numpy()[0, 0])

        is_genuine = score >= threshold
        return jsonify({
            "score": round(score, 4),
            "threshold": threshold,
            "result": "genuine" if is_genuine else "forgery",
            "result_cn": "真签名" if is_genuine else "伪签名"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        for tmp_path in tmp_files:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == '__main__':
    print("预加载模型...")
    _load_model()
    print("API 服务启动: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
