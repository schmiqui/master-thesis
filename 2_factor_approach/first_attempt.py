import random

import networkx as nx
import matplotlib.pyplot as plt
# from random_k_g_non_ham_graph_generator import best_regular_graph_near_tree


def get_clean_2_factor(G):
    """
    Finds a 2-factor by finding a perfect matching and taking its complement.
    This ensures the 2-factor consists of cycles present in G (length >= girth).
    """
    # Find a perfect matching
    # max_weight_matching with maxcardinality=True finds a perfect matching in k-regular graphs
    matching = nx.complementary_edges = nx.maximal_matching(G)
    # For a guaranteed perfect matching in k-regular bridgeless graphs:
    m = nx.max_weight_matching(G, maxcardinality=True)

    TF = nx.Graph()
    TF.add_nodes_from(G.nodes())
    # The 2-factor is all edges NOT in the perfect matching
    for u, v in G.edges():
        if (u, v) not in m and (v, u) not in m:
            TF.add_edge(u, v)

    return TF, m


def lift_to_hamiltonian(G):
    n_nodes = list(G.nodes())
    n = len(n_nodes)
    # Map nodes to 0..n-1 to avoid index collisions
    mapping = {node: i for i, node in enumerate(n_nodes)}
    G_mapped = nx.relabel_nodes(G, mapping)

    # 1. Find a 2-factor that inherits G's girth
    TF, matching = get_clean_2_factor(G_mapped)
    cycles = [list(c) for c in nx.connected_components(TF)]

    # 2. Initialize all edges as 'parallel'
    lifted_edges = {tuple(sorted(edge)): 'parallel' for edge in G_mapped.edges()}

    # 3. Cross exactly one edge per cycle in the 2-factor
    # This turns each cycle of length L into one cycle of length 2L
    for cycle_nodes in cycles:
        u = cycle_nodes[0]
        v = list(TF.neighbors(u))[0]
        lifted_edges[tuple(sorted((u, v)))] = 'cross'

    # 4. Merge these 2L-length cycles using edges from the Matching
    if len(cycles) > 1:
        Meta = nx.Graph()
        # Nodes in Meta are indices of cycles
        # Edges in Meta are edges in the matching that connect different cycles
        for (u, v) in matching:
            c_u = next(i for i, c in enumerate(cycles) if u in c)
            c_v = next(i for i, c in enumerate(cycles) if v in c)
            if c_u != c_v:
                Meta.add_edge(c_u, c_v, edge=(u, v))

        # Use a Spanning Tree to connect all cycle-components
        tree = nx.minimum_spanning_tree(Meta)
        for _, _, data in tree.edges(data=True):
            u_g, v_g = data['edge']
            lifted_edges[tuple(sorted((u_g, v_g)))] = 'cross'

    # 5. Build the Hamiltonian Graph H
    H = nx.Graph()
    for (u, v), edge_type in lifted_edges.items():
        if edge_type == 'parallel':
            H.add_edge(u, v)
            H.add_edge(u + n, v + n)
        else:
            H.add_edge(u, v + n)
            H.add_edge(v, u + n)

    return H


def verify_hamiltonian(G):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    if n == 0: return False
    start_node = nodes[0]
    path = [start_node]
    visited = {start_node}

    def backtrack(curr):
        if len(path) == n:
            return G.has_edge(curr, start_node)
        for neighbor in G.neighbors(curr):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                if backtrack(neighbor): return True
                path.pop()
                visited.remove(neighbor)
        return False

    return backtrack(start_node)


# --- Test with your graph ---
graph_text = """
0: 115  282  126
1: 142  267  118
2: 118  275  136
3: 128  259  137
4: 129  272  112
5: 130  287  100
6: 137  270  129
7: 123  279  131
8: 135  280  105
9: 120  242  133
10: 131  249  142
11: 105  261  123
12: 133  285  107
13: 138  273  125
14: 139  274  109
15: 140  283   96
16: 102  271  111
17: 112  281  138
18: 125  256  115
19: 109  244  140
20: 136  286  117
21: 98  250  141
22: 106  251  119
23: 134  252  104
24: 117  262  135
25: 141  263  120
26: 119  276   98
27: 104  260  122
28: 143  284   99
29: 113  246  124
30: 114  257  101
31: 126  269  128
32: 96  253  143
33: 99  240  130
34: 124  255  114
35: 107  264  132
36: 108  265  116
37: 121  266   97
38: 132  277  106
39: 116  241  134
40: 97  247  108
41: 127  258  110
42: 100  243  139
43: 101  268  127
44: 110  245  102
45: 122  278  103
46: 103  248  121
47: 111  254  113
48: 140  339  173
49: 121  356  179
50: 119  360  190
51: 143  340  174
52: 130  349  185
53: 114  350  147
54: 110  353  176
55: 122  337  153
56: 134  343  165
57: 135  371  189
58: 132  338  171
59: 133  346  183
60: 97  374  168
61: 139  336  160
62: 127  351  148
63: 102  364  157
64: 129  365  158
65: 111  354  177
66: 124  367  186
67: 125  368  187
68: 108  344  166
69: 136  345  167
70: 142  357  180
71: 106  381  164
72: 141  347  184
73: 103  348  146
74: 104  361  154
75: 105  382  182
76: 113  341  144
77: 138  366  159
78: 115  377  172
79: 101  342  163
80: 126  369  188
81: 137  378  191
82: 99  379  161
83: 123  358  181
84: 107  359  145
85: 120  372  151
86: 116  362  155
87: 117  363  156
88: 118  375  169
89: 128  352  149
90: 112  355  178
91: 100  380  162
92: 109  383  175
93: 98  373  152
94: 131  376  170
95: 96  370  150
96: 15   32   95
97: 37   40   60
98: 26   21   93
99: 28   33   82
100: 5   42   91
101: 30   43   79
102: 44   16   63
103: 45   46   73
104: 23   27   74
105: 8   11   75
106: 38   22   71
107: 12   35   84
108: 40   36   68
109: 14   19   92
110: 41   44   54
111: 16   47   65
112: 4   17   90
113: 47   29   76
114: 34   30   53
115: 18    0   78
116: 36   39   86
117: 20   24   87
118: 1    2   88
119: 22   26   50
120: 25    9   85
121: 46   37   49
122: 27   45   55
123: 11    7   83
124: 29   34   66
125: 13   18   67
126: 0   31   80
127: 43   41   62
128: 31    3   89
129: 6    4   64
130: 33    5   52
131: 7   10   94
132: 35   38   58
133: 9   12   59
134: 39   23   56
135: 24    8   57
136: 2   20   69
137: 3    6   81
138: 17   13   77
139: 42   14   61
140: 19   15   48
141: 21   25   72
142: 10    1   70
143: 32   28   51
144: 261   76  199
145: 269   84  195
146: 282   73  223
147: 262   53  200
148: 275   62  212
149: 276   89  213
150: 286   95  216
151: 270   85  196
152: 281   93  205
153: 243   55  206
154: 259   74  198
155: 272   86  209
156: 287   87  234
157: 249   63  193
158: 277   64  214
159: 285   77  227
160: 247   61  228
161: 267   82  194
162: 279   91  202
163: 264   79  230
164: 256   71  192
165: 244   56  207
166: 253   68  220
167: 254   69  221
168: 273   60  210
169: 274   88  211
170: 283   94  224
171: 271   58  239
172: 263   78  201
173: 248   48  229
174: 260   51  237
175: 280   92  203
176: 242   54  204
177: 250   65  217
178: 278   90  238
179: 240   49  197
180: 255   70  222
181: 268   83  233
182: 284   75  225
183: 246   59  226
184: 257   72  235
185: 241   52  215
186: 251   66  218
187: 252   67  219
188: 265   80  231
189: 245   57  208
190: 258   50  236
191: 266   81  232
192: 312  164  362
193: 306  157  383
194: 301  161  341
195: 290  145  374
196: 298  151  348
197: 326  179  376
198: 289  154  381
199: 319  144  370
200: 294  147  379
201: 321  172  367
202: 288  162  350
203: 291  175  351
204: 316  176  352
205: 299  152  361
206: 300  153  382
207: 313  165  363
208: 334  189  347
209: 295  155  359
210: 296  168  372
211: 297  169  356
212: 305  148  380
213: 330  149  342
214: 307  158  353
215: 332  185  378
216: 292  150  364
217: 293  177  365
218: 302  186  366
219: 329  187  336
220: 314  166  375
221: 315  167  360
222: 327  180  338
223: 308  146  373
224: 309  170  337
225: 310  182  343
226: 311  183  371
227: 320  159  354
228: 335  160  355
229: 322  173  368
230: 303  163  377
231: 304  188  339
232: 317  191  340
233: 328  181  346
234: 323  156  344
235: 324  184  345
236: 325  190  357
237: 331  174  369
238: 318  178  349
239: 333  171  358
240: 179   33  289
241: 185   39  292
242: 176    9  294
243: 153   42  295
244: 165   19  296
245: 189   44  297
246: 183   29  299
247: 160   40  301
248: 173   46  288
249: 157   10  303
250: 177   21  305
251: 186   22  306
252: 187   23  307
253: 166   32  308
254: 167   47  309
255: 180   34  310
256: 164   18  311
257: 184   30  312
258: 190   41  290
259: 154    3  314
260: 174   27  291
261: 144   11  316
262: 147   24  293
263: 172   25  318
264: 163   35  319
265: 188   36  320
266: 191   37  321
267: 161    1  322
268: 181   43  323
269: 145   31  324
270: 151    6  325
271: 171   16  298
272: 155    4  326
273: 168   13  300
274: 169   14  328
275: 148    2  302
276: 149   26  329
277: 158   38  304
278: 178   45  330
279: 162    7  331
280: 175    8  332
281: 152   17  333
282: 146    0  313
283: 170   15  334
284: 182   28  315
285: 159   12  317
286: 150   20  335
287: 156    5  327
288: 202  248  360
289: 198  240  354
290: 195  258  349
291: 203  260  338
292: 216  241  346
293: 217  262  374
294: 200  242  337
295: 209  243  367
296: 210  244  342
297: 211  245  369
298: 196  271  336
299: 205  246  339
300: 206  273  364
301: 194  247  347
302: 218  275  348
303: 230  249  361
304: 231  277  382
305: 212  250  343
306: 193  251  344
307: 214  252  345
308: 223  253  353
309: 224  254  378
310: 225  255  355
311: 226  256  380
312: 192  257  340
313: 207  282  341
314: 220  259  350
315: 221  284  377
316: 204  261  362
317: 232  285  363
318: 238  263  375
319: 199  264  356
320: 227  265  357
321: 201  266  358
322: 229  267  359
323: 234  268  368
324: 235  269  383
325: 236  270  370
326: 197  272  351
327: 222  287  352
328: 233  274  365
329: 219  276  376
330: 213  278  371
331: 237  279  372
332: 215  280  373
333: 239  281  379
334: 208  283  366
335: 228  286  381
336: 298   61  219
337: 294   55  224
338: 291   58  222
339: 299   48  231
340: 312   51  232
341: 313   76  194
342: 296   79  213
343: 305   56  225
344: 306   68  234
345: 307   69  235
346: 292   59  233
347: 301   72  208
348: 302   73  196
349: 290   52  238
350: 314   53  202
351: 326   62  203
352: 327   89  204
353: 308   54  214
354: 289   65  227
355: 310   90  228
356: 319   49  211
357: 320   70  236
358: 321   83  239
359: 322   84  209
360: 288   50  221
361: 303   74  205
362: 316   86  192
363: 317   87  207
364: 300   63  216
365: 328   64  217
366: 334   77  218
367: 295   66  201
368: 323   67  229
369: 297   80  237
370: 325   95  199
371: 330   57  226
372: 331   85  210
373: 332   93  223
374: 293   60  195
375: 318   88  220
376: 329   94  197
377: 315   78  230
378: 309   81  215
379: 333   82  200
380: 311   91  212
381: 335   71  198
382: 304   75  206
383: 324   92  193
"""


def parse_adj(text):
    G = nx.Graph()
    for line in text.strip().splitlines():
        u, neighbors = line.split(":")
        for v in neighbors.split():
            G.add_edge(int(u), int(v))
    return G

# if __name__ == '__main__':
#
#     generated = True
#
#     G_base = None
#     if generated:
#         G_base = random_sparse_regular_graph_with_report(21, 10)
#     else:
#      G_base = parse_adj(graph_text)
#     k = d = 2 * G_base.number_of_edges() / G_base.number_of_nodes()
#     H_result = lift_to_hamiltonian(G_base)
#
#     print(f"Original Nodes: {G_base.number_of_nodes()}")
#     print(f"Original Girth: {nx.girth(G_base)}")
#     print(f"Result Nodes:   {H_result.number_of_nodes()}")
#     print(f"Result Girth:   {nx.girth(H_result)}")
#     print("=" * 30)
#     print("STABILITY")
#     print(f"Girth:          {nx.girth(H_result) == nx.girth(G_base)}")
#     print(f"Regularity:     {all(d == k for _, d in H_result.degree())}")
#     print(f"Hamiltonian:    {verify_hamiltonian(H_result)}")
#
#     print("=" * 30)
#     nx.draw_circular(H_result, with_labels=True, node_color='orange')
#     plt.show()


if __name__ == '__main__':
    attempts = 20
    n_max = 100
    k = 9
    hamiltonicity_threshold = 18
    for _ in range(attempts):
        n = random.randint(10, n_max)
        G_base = best_regular_graph_near_tree(n, k)

        k  = 2 * G_base.number_of_edges() / G_base.number_of_nodes()
        H_result = lift_to_hamiltonian(G_base)

        print(f"Original Nodes: {G_base.number_of_nodes()}")
        print(f"Original Girth: {nx.girth(G_base)}")
        print(f"Result Nodes:   {H_result.number_of_nodes()}")
        print(f"Result Girth:   {nx.girth(H_result)}")
        print("==" * 30)
        print("STABILITY")
        print(f"Girth:          {nx.girth(H_result) == nx.girth(G_base)}")
        print(f"Regularity:     {all(d == k for _, d in H_result.degree())}")
        if n <= hamiltonicity_threshold:
            print(f"Hamiltonian:    {verify_hamiltonian(H_result)}")
            nx.draw_circular(H_result, with_labels=True, node_color='orange')
            plt.show()
        print("__" * 30)



