# ostle_solver

[![DOI](https://zenodo.org/badge/405170430.svg)](https://zenodo.org/badge/latestdoi/405170430)

The Japanese description is after the English one. (日本語の説明文は英語版の後ろにあります)

### How to solve

```Shell
$ make
$ ./solver -r -b -i -p -1
```

#### options
- -r: execure retrograde analysis.
- -b: execute bfs.
- -i: enumerate interesting states.
- -p [n]: Compute n threads in parallel; if n=-1, all logical cores are used. Default is n=1 (=no parallelization).
- -t: execute unittests.
- -s [n]: The total number of pieces is restricted to a phase with n or less. If not specified, n=10 (= complete analysis).

The program performs a retrograde analysis and/or bfs, and writes the results as text files.

(each file is less than 50 MB, about 213 GB in total)

Note: solver uses AMD64 AVX2 and BMI2, so it will only work on Haswell or later for Intel CPUs and Excavator or later for AMD CPUs. Additionally, it makes heavy use of BMI2 pdep/pext instructions, therefore it may take a long time to execute on Zen2 and earlier generation AMD CPUs.

#### about python scripts

Python scripts require the full analysis files for the retrograde analysis and bfs results. They perform additional analysis. They works with PyPy 7.3.1 (Python 3.6.9).

- query.py: Receives state information via command line arguments and outputs the following:
  - The result of the retrograde analysis for the argument (win/loss and number of moves to the end of the game)
  - Results of bfs for the received phase
  - -s: The results of the retrograde analysis for all possible moves from the received game to the end of the game and for all legal moves along the way.

#### how to use query.py

For example, when the current board is 00100-11200-00001-31022-20020 and the previous board is 00000-22200-00102-32011-10010, in order to obtain the results of the retrograde analysis:

```Shell
$ python3 query.py 00100-11200-00001-31022-20020 -p 00000-22200-00102-32011-10010 -s
```

The above is sufficient. If you wish to know about a game in which no restricted move exist, you can omit it as shown below.

```Shell
$ python3 query.py 00100-11200-00001-31022-20020 -s
```


### 以下は日本語

https://gamemarket.jp/game/76310

※Ostle（オストル）は 雅ゲームスにより制作されたボードゲームです。当リポジトリと雅ゲームスとの関係はありません。

cf:
http://h1.hara.net.it-chiba.ac.jp/ostle/

※上記のwebサイトの作者様も当リポジトリと関係はありません。

### 完全解析の手順

```Shell
$ make
$ ./solver -r -b -i -p -1
```

#### solverのオプション
- -r: 後退解析を実行する。
- -b: 幅優先探索を実行する。
- -i: 面白い性質を持つ局面を列挙する。
- -p [n]: nスレッド並列で計算する。n==-1だと論理コア全て使う。指定しないとn=1（＝並列化されない）となる。
- -t: 単体テストを実行する。
- -s [n]: 駒の総数がn個以下の局面に限定する。指定しないとn=10（＝完全解析）となる。

solverが後退解析と幅優先探索を行うと、そのディレクトリに解析結果をテキストファイルで書き出す。

（各ファイル 50 MB 以下で、合計 213 GB くらい）

※注意：solverはAMD64のAVX2およびBMI2を使っているので、Intel CPUならHaswell世代以降、AMD CPUならExcavator世代以降でないと動作しない。また、BMI2のpdep/pext命令を多用しているので、Zen2世代およびそれ以前の世代のAMD CPUでは実行時間が長くかかるかもしれない。

#### pythonスクリプトについて

後退解析と幅優先探索の完全解析結果のファイルを読み取って、追加の解析を行うスクリプトが用意されている。PyPy 7.3.1 (Python 3.6.9) で動作確認した。

- query.py: コマンドライン引数で局面情報を受け取って、以下を出力する：
  - 受け取った局面に関する後退解析の結果（勝敗と決着までの手数）
  - 受け取った局面に関する幅優先探索の結果（初期局面から最短何手で到達できるか）
  - -s: 受け取った局面から決着までの読み筋すべてと、その途中における全合法手に関する後退解析の結果
- check（～～）.py: 名前の通り。

#### query.pyの使用例

現在の盤面が00100-11200-00001-31022-20020で、直前の盤面が00000-22200-00102-32011-10010であるときの、決着までの読み筋全てと途中における全合法手に関する後退解析の結果を得るには：

```Shell
$ python3 query.py 00100-11200-00001-31022-20020 -p 00000-22200-00102-32011-10010 -s
```

とすればよい。禁じ手が存在しない局面について知りたい場合は、以下に示すように直前の盤面の情報を省略してもよい。

```Shell
$ python3 query.py 00100-11200-00001-31022-20020 -s
```

### その他

開発中に試したけど実を結ばなかった高速化技法たちについて

2021/10/23:追記:以下で説明する方法たちは、最初のreleaseバージョンではすべて実装していたが、性能上の利点が見当たらなかったものについてはv1.0以降では削除されているか使われていない。

#### 盤面から添字を逆引きする2通りの方法について

後退解析の前に盤面の全列挙を行うわけだが、その全盤面たちが格納されている配列を仮にAとする。盤面pから手を指すことで別の盤面p'に遷移したとき、添字アクセスするために、配列A上でのp'の添字i（すなわち A[i] == p' なるi）を求める必要が生じる。それについてこのプログラムでは2通りの方法が実装されていて、好きに選べるようになっている。ひとつはcode2index関数で、配列Aがソート済みであることを利用して二分探索でiを求めるものである。時間計算量はO(log2(|A|))である。もう一つはfind関数で、配列Aと予備データを合わせてハッシュテーブルになるように予め準備したうえで、iをO(1)で求めることができる。

はじめに二分探索のcode2index関数を実装してパフォーマンスプロファイリングした結果、計算時間の大部分がこの逆引きに費やされていた。二分探索よりもハッシュテーブルのほうが速いと予想してそれも実装したのだが、実際に(4,4)の盤面のみに限定して後退解析した結果、二分探索のほうが20~40%ほど速かった。考えられる理由として、二分探索ではキャッシュヒット率が高いのかもしれない。配列Aがbitboardを整数とみなしてソートしてあることにより、似た盤面が配列A上で近くに配置されがちなのかもしれない。だとすると、同じ盤面pから異なる合法手で遷移した先の盤面p'たちもA上で近くに配置されがちなのかもしれない。一方でハッシュテーブルにおいてはpもp'も全てランダムに配置されているので、ほぼ毎回メインメモリを読みに行く必要があると考えられる。物理4コア論理8コアのCore-i7で、並列化すると二分探索版で5倍・ハッシュテーブル版で6倍くらい高速化されるのだが、このこともキャッシュヒット率が低いことを示唆している。

とはいえ、(4,4)の盤面のみに限定せず本当の全盤面の後退解析を行うときには、O(1)で処理できるハッシュテーブルのほうが速いという可能性はまだ残っている。（まだ試していないので）

ハッシュテーブル方式のもう一つのデメリットとして、メモリ消費量がハッシュテーブル方式だとおよそ50%増しになる。load factorが70%くらいまでしか上げられないことと予備データが必要であることの2つが原因である。

#### 配列を並べ替えて二分探索をキャッシュフレンドリーにする方法について

https://postd.cc/cache-friendly-binary-search/

ソート済み配列を二分探索するとき、アクセスする添字は前後に飛ぶことになる。ところで、任意のiについて、i番目の配列要素が二分探索中にアクセスされるとき、その瞬間にその二分探索のwhile文がn回ループしていたとして、そのnはiから一意に定まる。つまりf(i)=nなる関数fが存在するわけだが、f(i)が小さい配列要素を配列の前方に固めておくことで二分探索をよりキャッシュフレンドリーにできるということが知られている。（上のURLの記事がそれ）

並べ替えの方法の例としては、配列を値そのもので昇順にソートしたあとで「二分探索でn段目にアクセスされる要素」のnで昇順に安定ソートすればいい。nを求めるためには、仮想的に数列を平衡二分探索木とみなして深さ優先でトラバースするとよい。通りがけに数列の要素とレベルの関係を得られる。以下は配列の各要素が55bit以下であるとの仮定のもとで実際に動くコードである。（配列の要素数が2^n-1で表せない場合、平衡二分探索木が完全二分木にならないことに注意して実装する必要がある。上の記事内の例では都合よく31要素になっていたが……）

```
void dfs_binary_tree(const uint64_t index, const uint64_t level, uint64_t &counter, std::vector<uint64_t> &v) {

	if (index * 2 + 1 < v.size()) {
		dfs_binary_tree(index * 2 + 1, level + 1, counter, v);
	}

	v[counter++] |= level << 55;

	if (index * 2 + 2 < v.size()) {
		dfs_binary_tree(index * 2 + 2, level + 1, counter, v);
	}
}

void shuffle_sorted_to_levelwise(std::vector<uint64_t> &v) {
	//vが55bitで収まっていると仮定して、levelwiseに並べ替える。

	std::sort(v.begin(), v.end());
	assert(v.back() < (1ULL << 55));

	uint64_t counter = 0;
	dfs_binary_tree(0, 1, counter, v);
	assert(counter == v.size());

	std::sort(v.begin(), v.end());

	for (uint64_t i = 0; i < v.size(); ++i) {
		v[i] &= (1ULL << 55) - 1ULL;
	}
}
```

二分探索では、配列を平衡二分探索木(0-origin)とみなして根から葉に向けて見ていくやつを単にやればいい。（以下の通り）

```
	uint64_t code2index_levelwise(const uint64_t c, const std::vector<uint64_t> &v) {
		//vのどれかの値cを引数にとり、v[i]=cなるiを探して返す。
		//vが昇順にソートされてからlevelwiseに並べ替えられていると仮定して二分探索で求める。

		uint64_t i = 0;
		while (v[i] != c) {
			if (v[i] < c) {
				i = i * 2 + 2;
			}
			else {
				i = i * 2 + 1;
			}
			assert(i < v.size());
		}
		return i;
	}
```

配列へのアクセス順序が一様ランダムなテストケースにおいて、このキャッシュフレンドリーにする方法をとることで2倍くらい高速化されることを確認した。しかし、Ostleの完全解析のコードに実際に組み込んで試したところ、むしろ数割遅くなった。そもそもアクセスする配列要素が偏っているのが原因と推測される。

### その他（undo関数の必要性について）

盤面pのポインタと指し手mを引数に取り、mを指して盤面をp'に更新する関数をdo関数と呼ぶ。逆に、p'のポインタとmを引数に取りp'をpに戻す関数をundo関数と呼ぶ。

将棋ソフトとかではこのように1個の盤面データを破壊的変更しながら探索するのが普通である。そのような方式を取る理由はそのほうが速いからだが、背景事情として (1) pのデータ構造が大きい (2) 手を指すことによる盤面の変更が局所的である (3) 評価関数を差分評価するときの差分更新も(un)do関数のタイミングで行う などがある。

今回はundo関数を実装してしまったが、結局使わなかった。全列挙するだけなので差分評価とかは無いわけだが、そういう背景の違いを検討せずになんとなく実装してしまったのだった。
