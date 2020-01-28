# RNN网络生成随机名字


```python
# COPYRIGHT BY EX10SI0N @ www.github.com/Ex10si0n
```

## 数据采集处理


```python
from __future__ import absolute_import, division, print_function, unicode_literals
try:
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

import numpy as np
import os
import time
```


```python
# 处理读入文件为str类型
source = open("namesrc.txt").read()
```


```python
# 处理source中的unique
source_set = sorted(set(source))
```


```python
# 创建从非重复字符到索引的映射
char2idx = {u:i for i, u in enumerate(source_set)}
idx2char = np.array(source_set)
text_as_int = np.array([char2idx[c] for c in source])
```


```python
# 索引表打印测试
print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')
```

    {
      ' ' :   0,
      '业' :   1,
      '东' :   2,
      '久' :   3,
      '义' :   4,
      '乐' :   5,
      '云' :   6,
      '亚' :   7,
      '亦' :   8,
      '京' :   9,
      '亭' :  10,
      '仁' :  11,
      '仑' :  12,
      '仟' :  13,
      '仲' :  14,
      '伟' :  15,
      '伦' :  16,
      '佐' :  17,
      '佑' :  18,
      '佳' :  19,
      ...
    }


## 创建训练样本和目标


```python
seq_length = 2
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
for i in char_dataset.take(5):
    print(idx2char[i.numpy()])
```

    华
    君
     
    子
    睿



```python
seq = char_dataset.batch(seq_length + 1, drop_remainder = True)
for item in seq.take(5):
    print(repr(''.join(idx2char[item.numpy()])))
```

    '华君 '
    '子睿 '
    '宏君 '
    '旦宇 '
    '甫宇 '



```python
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = seq.map(split_input_target)
```


```python
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
```

    Input data:  '华君'
    Target data: '君 '



```python
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
```

    Step    0
      input: 49 ('华')
      expected output: 57 ('君')
    Step    1
      input: 57 ('君')
      expected output: 0 (' ')



```python
# 设定批和缓冲区大小
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset
```




    <BatchDataset shapes: ((64, 2), (64, 2)), types: (tf.int64, tf.int64)>



## 创建模型


```python
set_size = len(source_set)
embedding_dim = 256
rnn_units = 1024
```


```python
def build_model(set_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(set_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(set_size)
  ])
  return model
```


```python
model = build_model(
  set_size = len(source_set),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)
```


```python
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, set_size)")
```

    (64, 2, 476) # (batch_size, sequence_length, set_size)



```python
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (64, None, 256)           121856    
    _________________________________________________________________
    gru_2 (GRU)                  (64, None, 1024)          3938304   
    _________________________________________________________________
    dense_2 (Dense)              (64, None, 476)           487900    
    =================================================================
    Total params: 4,548,060
    Trainable params: 4,548,060
    Non-trainable params: 0
    _________________________________________________________________



```python
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
```


```python
sampled_indices
```




    array([403,  30])




```python
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())
```

    Prediction shape:  (64, 2, 476)  # (batch_size, sequence_length, vocab_size)
    scalar_loss:       6.1663694



```python
model.compile(optimizer='adam', loss=loss)
```

## 配置检查点


```python
# 检查点保存至的目录
checkpoint_dir = './training_checkpoints'

# 检查点的文件名
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
```


```python
EPOCHS=50
```


```python
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
```

    Train for 81 steps
    Epoch 1/10
    81/81 [==============================] - 5s 60ms/step - loss: 4.2397
    Epoch 2/10
    81/81 [==============================] - 4s 48ms/step - loss: 2.6707
    Epoch 3/10
    81/81 [==============================] - 4s 48ms/step - loss: 2.5728
    Epoch 4/10
    81/81 [==============================] - 4s 47ms/step - loss: 2.4364
    Epoch 5/10
    81/81 [==============================] - 4s 48ms/step - loss: 2.3159
    Epoch 6/10
    81/81 [==============================] - 4s 47ms/step - loss: 2.2223
    Epoch 7/10
    81/81 [==============================] - 4s 48ms/step - loss: 2.1524
    Epoch 8/10
    81/81 [==============================] - 4s 48ms/step - loss: 2.0929
    Epoch 9/10
    81/81 [==============================] - 4s 48ms/step - loss: 2.0519
    Epoch 10/10
    81/81 [==============================] - 4s 48ms/step - loss: 2.0309



```python
tf.train.latest_checkpoint(checkpoint_dir)
```




    './training_checkpoints/ckpt_10'




```python
model = build_model(set_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (1, None, 256)            121856    
    _________________________________________________________________
    gru_1 (GRU)                  (1, None, 1024)           3938304   
    _________________________________________________________________
    dense_1 (Dense)              (1, None, 476)            487900    
    =================================================================
    Total params: 4,548,060
    Trainable params: 4,548,060
    Non-trainable params: 0
    _________________________________________________________________



```python
def generate_text(model, start_string):
  num_generate = 1000
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = []

  temperature = 3.0

  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)

      predictions = tf.squeeze(predictions, 0)
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))
```

## 优化输出


```python
import re
name_arr = generate_text(model, start_string=u"卓")
name_arr = name_arr.replace(" ","")
interval = re.compile('.{2}')
lastname = ' '.join(interval.findall(name_arr))
lastname = lastname.split()
```

## 姓氏处理（数据来自百家姓）
概率模型: `p(x) = e^(-x * (4 / 97))`


```python
import random, math, csv
surname = open("surnamesrc.txt").read()
surname = surname.split()
p = [math.exp(- x * (4 / 97)) for x in range(1, len(surname))]
row = random.random()
surname_set = []
for j in range(len(name_arr)):
    for i in range(len(surname) - 1):
        if p[i] <= row:
            surname_set.append(surname[i])
            row = random.random()
            break
```


```python
n = min(len(surname), len(lastname))
print("result: " + str(n) + " names generated")
```

    result: 386 names generated



```python
name = []
for i in range(n):
    name.append(surname[i] + lastname[i])
print(name)
```

    ['曾卓滨', '赵熙烨', '钱远白', '孙蓓礼', '李樨晗', '周昱熊', '吴尊炎', '郑菲聪', '王乐斌', '冯立军', '陈云学', '褚云祯', '卫泓亭', '蒋科弈', '沈铮秀', '韩棋诩', '杨瑾卓', '朱贤洲', '秦科封', '尤淡焱', '许泓莓', '何宝朗', '吕宗平', '施盛谦', '张希晗', '孔飘烁', '曹珝凯', '严辰祖', '华谦奎', '金纯峰', '魏翰厚', '陶俊翔', '姜琛凡', '戚钢翰', '谢飘翊', '邹晗军', '喻超歌', '柏洋亦', '水荫成', '窦睿斌', '章侦磊', '云浩麒', '苏嘉毅', '潘棱焱', '葛昱科', '奚天易', '范瑞键', '彭伟凡', '郎豫磊', '鲁伟宗', '韦景硕', '昌教斌', '马基端', '苗量坚', '凤希箫', '花泽烁', '方曦贞', '俞凯炜', '任凯孝', '袁飞琛', '柳琛育', '酆廷鑫', '鲍炜昱', '史伟常', '唐瀚晗', '费双政', '廉坤飞', '岑政超', '薛奥俊', '雷槿渤', '贺佐颜', '倪竿影', '汤亭珅', '滕云香', '殷翰辰', '罗基鸿', '毕伦枫', '郝涛鸿', '邬炳济', '安晨然', '常斌郎', '乐旬硕', '于科莫', '时悦珝', '傅俊林', '皮蕾珝', '卞格颜', '齐桦廷', '康杰雪', '伍俞乐', '余烨辉', '元淇伟', '卜岩仲', '顾易珂', '孟洋存', '平钧诩', '黄晗纹', '和璟羿', '穆贝翔', '萧风泓', '尹晨泓', '姚子化', '邵熙熠', '湛竿麒', '汪裕祖', '祁超常', '毛豫辰', '禹炜蓓', '狄麒麒', '米瑞然', '贝冬琮', '明聪骏', '臧聪俊', '计佐运', '伏绮郎', '成文祯', '戴豪轩', '谈谨涛', '宋祖瑜', '茅超成', '庞藏智', '熊龙翰', '纪凡仁', '舒斌迪', '屈先深', '项俊温', '祝君森', '董睿扬', '梁昱景', '杜迪桦', '阮风胥', '蓝莉大', '闵远郎', '席毅睿', '季飞东', '麻科贤', '强慧廷', '贾昕庄', '路基嗣', '娄珝泓', '危琳超', '江贤跃', '童毅博', '颜超洋', '郭键翔', '梅希金', '盛硕悠', '林仑鹤', '刁论郎', '徐里征', '邱衡俞', '骆镇仑', '高琮珝', '夏弈莉', '蔡仁成', '田君政', '樊秀侦', '胡辰厚', '凌叶筱', '霍贞熊', '虞煊萃', '万辉宗', '支翔辉', '柯化端', '昝贝羿', '管萃鸿', '卢琛奥', '莫廷泓', '经铮云', '房钢希', '裘力闻', '缪瑞桐', '干泓坤', '解喻麒', '应佑洋', '宗炜硕', '丁梓聪', '宣秀诚', '邓瑞彬', '贲悦景', '郁纹冬', '单昕超', '杭俊龙', '洪贝勇', '包洲倩', '诸剑磊', '左倬鸿', '石焱钰', '崔意仲', '吉伦正', '钮俊瑜', '龚祯亦', '程剑益', '嵇斌军', '邢瀚纹', '滑翊泓', '裴鸿豪', '陆业君', '荣旻智', '翁翰羽', '荀钧语', '羊佳樾', '於煜方', '惠政歌', '甄剑继', '麴家坤', '家竿苇', '封成钧', '芮祺钢', '羿翊琳', '储震方', '靳琛枚', '汲瑞芃', '邴涛意', '糜祯骏', '松鸿明', '段弈乐', '富铠诺', '巫征斌', '乌圣依', '焦卓叶', '巴贞纹', '弓胥盛', '牧彦焱', '隗桥军', '山雨宗', '谷云杰', '关渤聪', '妫棋卓', '桂宇韬', '国栐骏', '海杰慕', '侯希嘉', '后易成', '弘坚剑', '庚鹤简', '巩侦温', '贡天烁', '古俞权', '甘睿德', '郜潇曦', '戈宇希', '艾温琛', '文珩云', '丰伦维', '鄂晏乐', '法福智', '都珝天', '钭圩聪', '堵鹏泓', '扶拓珅', '符聪轩', '楚翔基', '从岩学', '苍亚箫', '池荣旦', '仇翊睿', '是韬俞', '党林奕', '东林弘', '白宗悦', '班熙政', '暴祺年', '敖弼烁', '能嘉剑', '聂毅方', '牛羽翊', '农珩磊', '薄俞祖', '步煜斌', '边淇意', '晁冬聪', '巢朗诺', '车贝凯', '查乐翊', '柴珝宇', '荆度度', '居毅声', '盖桥政', '蒯霖悦', '夔烁云', '利希磊', '连柏仑', '廖寿倩', '蔺权诩', '刘颜倩', '龙梁旷', '乜程盛', '蒙榕怀', '禄臻政', '劳翔俊', '满佐嘉', '那康英', '宓俊尊', '宁翊轩', '隆冬毅', '栾成桐', '终华翰', '仲跃森', '阎轩羽', '晏鸿倬', '燕睿犀', '翟华彦', '欧铭秀', '竺天聪', '訾鹏理', '祖瑞科', '越皓煊', '宰臻瀚', '黎嘉朗', '厉邦炜', '郦泓继', '景泽铮', '鞠森颜', '空子乐', '寇铠政', '匡森慕', '赖颜尚', '简科渭', '暨朗馨', '冀聚飘', '郏烜益', '姬乐进', '桓涛雄', '益祖焱', '仉臻寿', '仰纪薇', '养瑞杰', '伊谦鸣', '易虞轩', '鱼风渤', '雍纹嘉', '卓绅卓', '钟宁山', '印泽智', '叶依炜', '庄琛依', '旷富烨', '邝渤牧', '来煊超', '操方笑', '承子翰', '浮鸿朗', '洑熙翾', '蓬曦骏', '朋灏烁', '朴思杰', '詹冬翰', '游贞洲', '芈景天', '睦雨珝', '冒斌宗', '门小祖', '慕莓曦', '谬焱易', '麦骏熙', '南叶森', '区颜智', '苟政科', '光贝澜', '刚镇理', '归圩辰', '宦来灏', '荤阔侦', '户浩贞', '蹇祯灏', '翦淇天', '揭皓堇', '矫萃宁', '鹿瀚跃', '楼宗亭', '励诺阔', '商嘉庄', '尚旻弼', '韶京豪', '萨骏烁', '赛珂成', '桑本佐', '沙鹏胥', '璩悦政', '茹昊翎', '戎纯拓', '容琛泓', '融浩超', '曲剑先', '瞿力学']



```python

```
