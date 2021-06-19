#!/usr/bin/env python
# coding: utf-8

shead_anchors = ((18,), (36,), (72,),
				 (128,), (208,), (320,),
				 (512,), (768,))

chuman_anchors = ((10,), (48,), (128,), (192,), (256,), (320,), (512,))
comb_anchors = ((12,), (32,), (64,), (112,), (196, ), (256,), (384,), (512,))
hh_anchors = ((12,), (18,), (24,), (32,), (48, ), (64,), (128,))


sh_anchors  = {'anchor_sizes' : shead_anchors,
			  'aspect_ratios' : ((0.5, 1.0, 1.5),) * len(shead_anchors)}

ch_anchors = {'anchor_sizes' : chuman_anchors,
			  'aspect_ratios' : ((0.5, 1.0, 2.0),) * len(chuman_anchors)}

combined_anchors = {'anchor_sizes' : comb_anchors,
					'aspect_ratios' : ((0.5, 1.0, 1.5),) * len(comb_anchors)}

headhunt_anchors = {'anchor_sizes' : hh_anchors,
					'aspect_ratios' : ((0.5, 1.0, 1.5),) * len(hh_anchors)}