import os
__curr_path__ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

__model_path__ = os.path.join(__curr_path__, 'models')


# 公式相关的
layout = {
    'model_path': 'detection/yolov8n_cdla.onnx',
    'labels': ["Header",
                "Text",
                "Reference",
                "Figure caption",
                "Figure",
                "Table caption",
                "Table",
                "Title",
                "Footer",
                "Equation",
                ]
}
# det
det_text = {
    'model_path': 'detection/det_text.onnx',
    'config':{  "limit_side_len":960,
                "thresh": 0.3, 
                "box_thresh": 0.5, 
                "max_candidates": 1000, "unclip_ratio": 1.5, 
                "use_dilation": False, "score_mode": "fast", 
                "box_type": "quad"}
}
det_formula = {
    'model_path': 'detection/yolov8m_formula.onnx',
    'labels': ['Paragraph',
                'Head',
                'Footnote',
                'HeaderFooter',
                'Caption',
                'Table',
                'Figure',
                'Formula',
                ]
}


rec_formula = {
    'model_path': 'recognition/rec_formula',
    'config': {'max_width': 672, 
               'max_height': 192, 
               'min_height': 32, 
               'min_width': 32, 
               'bos_token': 1, 
               'max_seq_len': 512, 
               'eos_token': 2, 
               'temperature': 0.00001},
    'models': {'decoder':'decoder.onnx',
               'encoder':'encoder.onnx',
               'image_resizer':'image_resizer.onnx',
               'tokenizer':'tokenizer.json'}
}
rec_text = {
    'model_path': 'recognition/rec_text',
    'config': {'max_width': 672, 
               'max_height': 192, 
               'min_height': 32, 
               'min_width': 32, 
               'bos_token': 1, 
               'max_seq_len': 512, 
               'eos_token': 2, 
               'temperature': 0.00001},
    'names': {'model':'rec_text.onnx',
               'label':'rec_text.res'}
}

