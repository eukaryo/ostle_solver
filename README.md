# ostle_solver

https://gamemarket.jp/game/76310

※Ostle（オストル）は 雅ゲームスにより制作されたボードゲームです。当リポジトリと雅ゲームスとの関係はありません。

cf:
http://h1.hara.net.it-chiba.ac.jp/ostle/

※上記のwebサイトの作者様も当リポジトリと関係はありません。

ｶﾞﾊﾞｱﾙｶﾓ

## 局面の全列挙

まず最初に、初期局面から到達可能な局面を全列挙して、その総数（あるいはその上界）を求めた。

### 全列挙の定義

- 対称な局面は同一視する。（水平反転・垂直反転・行列転置の3操作をやるorやらないで合計8通りの対称性がある）
- 手番側が即座に勝利できる局面までを含み、その手を指して勝利した後の局面は含めない。
- 手番側が即座に勝利できる局面でも、その手を指さなくてもよいとする。言い換えると、どこかで勝利を見逃さないと決して辿り着けないような局面も含める。

### 考察

空白25マスに対して『最初に穴1個を配置して、残り24マスに手番側のコマ5or4個と相手のコマ5or4個を配置する』と解釈し、これの場合の数を考える。このとき対称な局面を同一視するならば、穴の配置箇所として考慮すべき場所は、下図で"o"になっている6マス（実装上の番号では0,1,2,6,7,12）だけでよい。なぜなら、その他のマスに穴を配置した場合、その後でコマをどのように配置しようとも、結果生じる盤面と対称な盤面が、列挙した盤面のなかに必ず存在するからである。

| 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| o | o | o | . | . |
| . | o | o | . | . |
| . | . | o | . | . |
| . | . | . | . | . |
| . | . | . | . | . |

もう一つのobservationとして、上記の6マスのうち別の穴に配置した2つの盤面同士は、決して対称にならない。また手番側or相手のコマの数が異なる盤面同士も決して対称にならない。ゆえに、『穴の位置とコマの数』の場合分け（24通り）によって、全局面を網羅しつつ相互排他的に分割できる。加えて、対称な局面の検出もその分割の内側同士でだけ考慮すればよくなる。

### 結果（局面数）

合計2,735,147,685局面だった。~~ただし、これは初期局面から到達不可能な局面を含む可能性がある。~~ （先手と後手が協力すれば任意の局面に到達できるはず。証明はしていないが、ほとんど間違いなくそうだと思っている）

| 穴の位置 | 手番側のコマの数 | 相手のコマの数 | 局面数 |
|---:|---:|---:|---:|
| 0 | 5 | 5 | 247127256 |
| 1 | 5 | 5 | 494236512 |
| 2 | 5 | 5 | 247127256 |
| 6 | 5 | 5 | 247127256 |
| 7 | 5 | 5 | 247127256 |
| 12 | 5 | 5 | 61788564 |
| 0 | 5 | 4 | 82378152 |
| 1 | 5 | 4 | 164745504 |
| 2 | 5 | 4 | 82378152 |
| 6 | 5 | 4 | 82378152 |
| 7 | 5 | 4 | 82378152 |
| 12 | 5 | 4 | 20598588 |
| 0 | 4 | 5 | 82378152 |
| 1 | 4 | 5 | 164745504 |
| 2 | 4 | 5 | 82378152 |
| 6 | 4 | 5 | 82378152 |
| 7 | 4 | 5 | 82378152 |
| 12 | 4 | 5 | 20598588 |
| 0 | 4 | 4 | 25744590 |
| 1 | 4 | 4 | 51482970 |
| 2 | 4 | 4 | 25744590 |
| 6 | 4 | 4 | 25744590 |
| 7 | 4 | 4 | 25744590 |
| 12 | 4 | 4 | 6438855 |

| 手番側のコマの数 | 相手のコマの数 | 局面数 |
|---:|---:|---:|
| 5 | 5 | 1544534100 |
| 5 | 4 | 514856700 |
| 4 | 5 | 514856700 |
| 4 | 4 | 160900185 |

## 後退解析

Ostleには千日手が存在し、有限ゲームではない。（二人零和有限確定完全情報ゲームの有限）

Ostleには『直前の盤面に戻すような指し手を禁じ手とする』というルールがある。これは一部の千日手を防ぐルールだが、このルールで防げない千日手が存在するためである。

このようなゲームを完全解析する手法として後退解析というものが存在する。後退解析という手法そのものの詳細の説明は割愛する。

### 考察

- 後退解析では、局面をノードとして指し手をエッジとするゲームグラフを考える。のだが、上述の『盤面の全列挙』で列挙した『盤面は』、後退解析の文脈における『ノード（＝局面）』ではない。『盤面』の情報に『禁じ手が存在するか、するならどの指し手か』の情報を加えると『ノード（＝局面）』になる。
- 各盤面における合法手の数は最大で24通りである。（着手位置6箇所×4方向）ゆえに、各盤面は最大25個のノードを内包している。
- メモリ使用量をできるだけ削減したい。そこで各ノードについて『最短あと何手で終局するか』の情報を求めるのを諦めるとすると、各ノードが後退解析中に保持すべき情報は2bitで足りる。すなわち(1)手番側の勝利が確定(2)手番側の敗北が確定(3)未確定　のどれであるかが保持できていればよい。よって、各盤面に64bit整数を割り当てれば解析できる。これを仮に『解析結果テーブル』と呼ぶ。
- メモリ使用量をできるだけ削減したい。そこで、ゲームグラフを陽に保持することなく後退解析することにする。この場合、大まかな手順としては以下の通りになる。

(1) 『解析結果テーブル』を全て未解析で初期化する。

(2) 全ての盤面に1つづつ注目する。仮にpに注目したとする。以下の(2-1)～(2-3)を行う。

　(2-1) pの合法手を全列挙し、それを指した先のノードの勝敗が解析済みかを調べる。

　(2-2) それで得た解析結果の情報に基づき、pが内包する各ノードの勝敗の解析を試みる。

　(2-3) 試みた解析結果を『解析結果テーブル』に上書きする。

(3) 直近の(2)によって『解析結果テーブル』が変更されなかったなら終了。少しでも変更されたなら(2)に戻る。

- 盤上のコマの数は一方的に減るだけで決して増えない。よって、両者のコマの数が(4,4)の盤面のみに限定しても、そのなかで後退解析が成立する。また、{(5,4),(4,5),(4,4)}の盤面のみに限定しても同様である。

# 結果

(4,4)の盤面のみに限定して後退解析ができた。（限定した理由は単に手元のPCのメモリ搭載量が不十分だからで、コードの機能としては全体の解析を行うこともできる。）（ｶﾞﾊﾞｱﾙｶﾓ）

## その他

### undo関数の必要性について

盤面pのポインタと指し手mを引数に取り、mを指して盤面をp'に更新する関数をdo関数と呼ぶ。逆に、p'のポインタとmを引数に取りp'をpに戻す関数をundo関数と呼ぶ。

将棋ソフトとかではこのように1個の盤面データを破壊的変更しながら探索するのが普通である。そのような方式を取る理由はそのほうが速いからだが、背景事情として (1) pのデータ構造が大きい (2) 手を指すことによる盤面の変更が局所的である (3) 評価関数を差分評価するときの差分更新も(un)do関数のタイミングで行う などがある。

今回はundo関数を実装してしまったが、結局使わなかった。全列挙するだけなので差分評価とかは無いわけだが、そういう背景の違いを検討せずになんとなく実装してしまったのだった。
