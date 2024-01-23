# Отчет: оптимизация параметров конфигурации для Triton Server
_Используется бекенд ONNX(CPU) см. tag [`hw2`](https://github.com/KamillaZyal/MLOps-HW/tree/hw2)_

## Описание решаемой задачи

Проект реализует модель на основе [pytorch-lightning](https://lightning.ai/) для классификации изображений цифр в наборе данных MNIST.

## Cистемная конфигурация

|Параметр|Значение|
|-|-|
OS | Windows 10 v.Домашняя + WSL2
CPU | Intel(R) Core(TM) i7-8550U CPU
vCPU | 8
RAM | 8 ГБ

## Описание стуктуры model_repository

```
model_repository/
└── mnist-onnx               # директория для данных модели mnist-onnx
    ├── 1                    # версия
    └── config.pbtxt         # конфигурация модели

2 directories, 1 file
```

## Оценка параметров оптимизации
Создание контейнера для утилиты SDK:

```
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.04-py3-sdk
```
Запуск тестирования:

```
perf_analyzer -m mnist-onnx -u localhost:8500 --concurrency-range 1:8 --shape input.1:1,1,28,28
```

До оптимизации:

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 799.089 infer/sec, latency 1250 usec
Concurrency: 2, throughput: 1315.39 infer/sec, latency 1518 usec
Concurrency: 3, throughput: 1539.2 infer/sec, latency 1947 usec
Concurrency: 4, throughput: 1569.26 infer/sec, latency 2547 usec
Concurrency: 5, throughput: 1531.98 infer/sec, latency 3262 usec
Concurrency: 6, throughput: 1548.98 infer/sec, latency 3872 usec
Concurrency: 7, throughput: 1587.38 infer/sec, latency 4407 usec
Concurrency: 8, throughput: 1581.72 infer/sec, latency 5055 usec
```
### Оценка параметра instance_groups

**Оценим влияние количества vCPU, варируя параметр `count`**

Результаты для `count: 8`. Наблюдаются негативные изменения в отличие от default параметров (`troughput` уменьшилась, `latency` увеличилась).

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 496.494 infer/sec, latency 2012 usec
Concurrency: 2, throughput: 765.182 infer/sec, latency 2614 usec
Concurrency: 3, throughput: 861.178 infer/sec, latency 3481 usec
Concurrency: 4, throughput: 1024.3 infer/sec, latency 3901 usec
Concurrency: 5, throughput: 1082.82 infer/sec, latency 4619 usec
Concurrency: 6, throughput: 825.515 infer/sec, latency 7268 usec
Concurrency: 7, throughput: 1102.75 infer/sec, latency 6346 usec
Concurrency: 8, throughput: 1115.81 infer/sec, latency 7181 usec
```

Результаты для `count: 4`. Показатели улучшились (`troughput` увеличилась, `latency` уменьшилась).

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 512.471 infer/sec, latency 1949 usec
Concurrency: 2, throughput: 916.101 infer/sec, latency 2182 usec
Concurrency: 3, throughput: 1112.49 infer/sec, latency 2692 usec
Concurrency: 4, throughput: 1250.34 infer/sec, latency 3200 usec
Concurrency: 5, throughput: 1342.92 infer/sec, latency 3721 usec
Concurrency: 6, throughput: 1414 infer/sec, latency 4241 usec
Concurrency: 7, throughput: 1475.22 infer/sec, latency 4743 usec
Concurrency: 8, throughput: 1511.4 infer/sec, latency 5288 usec
```

Результаты для `count: 2`. Положительные изменения сохраняются с уменьшением count.

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 805.388 infer/sec, latency 1228 usec
Concurrency: 2, throughput: 1131.89 infer/sec, latency 1765 usec
Concurrency: 3, throughput: 1379.8 infer/sec, latency 2172 usec
Concurrency: 4, throughput: 1485.89 infer/sec, latency 2689 usec
Concurrency: 5, throughput: 1443.72 infer/sec, latency 3461 usec
Concurrency: 6, throughput: 1519.57 infer/sec, latency 3946 usec
Concurrency: 7, throughput: 1475.15 infer/sec, latency 4743 usec
Concurrency: 8, throughput: 1479.9 infer/sec, latency 5403 usec
```
Результаты для `count: 1`. Лучший результат.

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1210.41 infer/sec, latency 825 usec
Concurrency: 2, throughput: 2208.91 infer/sec, latency 904 usec
Concurrency: 3, throughput: 2277.75 infer/sec, latency 1315 usec
Concurrency: 4, throughput: 2230.15 infer/sec, latency 1792 usec
Concurrency: 5, throughput: 2202.13 infer/sec, latency 2269 usec
Concurrency: 6, throughput: 2219.43 infer/sec, latency 2702 usec
Concurrency: 7, throughput: 2164.3 infer/sec, latency 3232 usec
Concurrency: 8, throughput: 2192.86 infer/sec, latency 3646 usec
```

 #### **Выводы**

При `count: 1` для *instance_groups* были зафиксированы наилучшие значения показателей (`troughput` наибольшее, а `latency` наименьшее).

### Delayed Batching
**При экспорте было задано, что модель .onnx обрабатывает данные size = [1, 1, 28, 28] -> параметр max_batch_size не анализируем.**
**Оценим влияние параметра времени задержки запроса, варируя `max_queue_delay_microseconds` (максимальное время задержки запроса в микросекунд).**

Результаты для `max_queue_delay_microseconds: 2000`. Наблюдаются негативные изменения в отличие от default параметров (`troughput` уменьшилась, `latency` увеличилась).

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1073.63 infer/sec, latency 930 usec
Concurrency: 2, throughput: 2210.79 infer/sec, latency 903 usec
Concurrency: 3, throughput: 2271.1 infer/sec, latency 1319 usec
Concurrency: 4, throughput: 2206.72 infer/sec, latency 1811 usec
Concurrency: 5, throughput: 2223.93 infer/sec, latency 2246 usec
Concurrency: 6, throughput: 2231.91 infer/sec, latency 2687 usec
Concurrency: 7, throughput: 2210.21 infer/sec, latency 3165 usec
Concurrency: 8, throughput: 2220.75 infer/sec, latency 3600 usec
```

Результаты для `max_queue_delay_microseconds: 500`. Наблюдаются улучшения значений (`troughput` увеличилась, `latency` уменьшилась).

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1052.73 infer/sec, latency 948 usec
Concurrency: 2, throughput: 2028.4 infer/sec, latency 984 usec
Concurrency: 3, throughput: 2037.83 infer/sec, latency 1470 usec
Concurrency: 4, throughput: 1916.98 infer/sec, latency 2085 usec
Concurrency: 5, throughput: 2144.4 infer/sec, latency 2330 usec
Concurrency: 6, throughput: 2148.51 infer/sec, latency 2791 usec
Concurrency: 7, throughput: 2114.79 infer/sec, latency 3308 usec
Concurrency: 8, throughput: 2085.93 infer/sec, latency 3833 usec
```

Результаты для `max_queue_delay_microseconds: 50`. Положительная тенденция сохраняется.

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1213.18 infer/sec, latency 822 usec
Concurrency: 2, throughput: 2107.12 infer/sec, latency 947 usec
Concurrency: 3, throughput: 2093.21 infer/sec, latency 1431 usec
Concurrency: 4, throughput: 2424.34 infer/sec, latency 1648 usec
Concurrency: 5, throughput: 2367.25 infer/sec, latency 2110 usec
Concurrency: 6, throughput: 2424.06 infer/sec, latency 2473 usec
Concurrency: 7, throughput: 2409.45 infer/sec, latency 2903 usec
Concurrency: 8, throughput: 2428.63 infer/sec, latency 3292 usec
```

Результаты для `max_queue_delay_microseconds: 40`. Присуствуют Warnings.
```
[WARNING] Perf Analyzer is not able to keep up with the desired load. The results may not be accurate.
Request concurrency: 8
Failed to obtain stable measurement within 10 measurement windows for concurrency 8. Please try to increase the --measurement-interval.
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1095.44 infer/sec, latency 911 usec
Concurrency: 2, throughput: 1835.88 infer/sec, latency 1088 usec
Concurrency: 3, throughput: 1873.46 infer/sec, latency 1599 usec
Concurrency: 4, throughput: 1915.4 infer/sec, latency 2086 usec
Concurrency: 5, throughput: 1870.3 infer/sec, latency 2672 usec
Concurrency: 6, throughput: 1922.37 infer/sec, latency 3119 usec
Concurrency: 7, throughput: 1904.29 infer/sec, latency 3674 usec
Concurrency: 8, throughput: 1904.29 infer/sec, latency 3674 usec
```
#### **Выводы**
При `max_queue_delay_microseconds: 50` для *dynamic_batching* были зафиксированы наилучшие значения показателей (`troughput` наибольшее, а `latency` наименьшее).
