# Отчет: оптимизация параметров конфигурации для Triton Server
_Используется бекенд ONNX(CPU) см. tag `hw2`_

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
Concurrency: 1, throughput: 287.695 infer/sec, latency 3473 usec
Concurrency: 2, throughput: 458.072 infer/sec, latency 4363 usec
Concurrency: 3, throughput: 514.742 infer/sec, latency 5824 usec
Concurrency: 4, throughput: 557.191 infer/sec, latency 7175 usec
Concurrency: 5, throughput: 570.163 infer/sec, latency 8751 usec
Concurrency: 6, throughput: 551.338 infer/sec, latency 10882 usec
Concurrency: 7, throughput: 549.609 infer/sec, latency 12727 usec
Concurrency: 8, throughput: 554.95 infer/sec, latency 14400 usec
```
#### Оценка параметра instance_groups

**Оценим влияние количества vCPU, варируя параметр `count`**

Результаты для `count: 8`. Наблюдаются негативные изменения в отличие от default параметров (`troughput` уменьшилась, `latency` увеличилась).

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 190.306 infer/sec, latency 5252 usec
Concurrency: 2, throughput: 305.644 infer/sec, latency 6541 usec
Concurrency: 3, throughput: 360.106 infer/sec, latency 8328 usec
Concurrency: 4, throughput: 364.975 infer/sec, latency 10950 usec
Concurrency: 5, throughput: 396.246 infer/sec, latency 12619 usec
Concurrency: 6, throughput: 428.991 infer/sec, latency 13980 usec
Concurrency: 7, throughput: 434.704 infer/sec, latency 16105 usec
Concurrency: 8, throughput: 454.478 infer/sec, latency 17592 usec
```

Результаты для `count: 4`. Показатели улучшились (`troughput` увеличилась, `latency` уменьшилась).

```bash
Concurrency: 1, throughput: 209.291 infer/sec, latency 4775 usec
Concurrency: 2, throughput: 352.205 infer/sec, latency 5677 usec
Concurrency: 3, throughput: 381.955 infer/sec, latency 7851 usec
Concurrency: 4, throughput: 419.499 infer/sec, latency 9525 usec
Concurrency: 5, throughput: 438.827 infer/sec, latency 11389 usec
Concurrency: 6, throughput: 476.559 infer/sec, latency 12583 usec
Concurrency: 7, throughput: 453.308 infer/sec, latency 12063 usec
Concurrency: 8, throughput: 488.308 infer/sec, latency 12741 usec
```

Результаты для `count: 2`. Положительные изменения сохраняются с уменьшением count.

```bash
Concurrency: 1, throughput: 292.456 infer/sec, latency 3417 usec
Concurrency: 2, throughput: 471.706 infer/sec, latency 4237 usec
Concurrency: 3, throughput: 568.031 infer/sec, latency 5278 usec
Concurrency: 4, throughput: 562.687 infer/sec, latency 7100 usec
Concurrency: 5, throughput: 639.398 infer/sec, latency 7818 usec
Concurrency: 6, throughput: 596.633 infer/sec, latency 10051 usec
Concurrency: 7, throughput: 584.104 infer/sec, latency 11981 usec
Concurrency: 8, throughput: 649.717 infer/sec, latency 12325 usec
```
Результаты для `count: 1`. Лучший результат.

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1409 infer/sec, latency 708 usec
Concurrency: 2, throughput: 2195.89 infer/sec, latency 909 usec
Concurrency: 3, throughput: 2117.92 infer/sec, latency 1415 usec
Concurrency: 4, throughput: 2270.15 infer/sec, latency 1760 usec
Concurrency: 5, throughput: 2202.08 infer/sec, latency 2269 usec
Concurrency: 6, throughput: 2078.64 infer/sec, latency 2885 usec
Concurrency: 7, throughput: 2180.04 infer/sec, latency 3209 usec
Concurrency: 8, throughput: 2184.33 infer/sec, latency 3661 usec
```

**Выводы**
При `count: 1` для *instance_groups* были зафиксированы наилучшие значения показателей (`troughput` наибольшее, а `latency` наименьшее).

#### Delayed Batching
**При экспорте было задано, что модель .onnx обрабатывает данные size = [1, 1, 28, 28] -> параметр max_batch_size не анализируем.**
**Оценим влияние параметра времени задержки запроса, варируя `max_queue_delay_microseconds` (максимальное время задержки запроса в микросекунд).**

Результаты для `max_queue_delay_microseconds: 2000`. Наблюдаются негативные изменения в отличие от default параметров (`troughput` уменьшилась, `latency` увеличилась).

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1303.47 infer/sec, latency 766 usec
Concurrency: 2, throughput: 1924.15 infer/sec, latency 1038 usec
Concurrency: 3, throughput: 1729.02 infer/sec, latency 1733 usec
Concurrency: 4, throughput: 1778.35 infer/sec, latency 2247 usec
Concurrency: 5, throughput: 2005.86 infer/sec, latency 2491 usec
Concurrency: 6, throughput: 2051.75 infer/sec, latency 2923 usec
Concurrency: 7, throughput: 2097.8 infer/sec, latency 3335 usec
Concurrency: 8, throughput: 2109.33 infer/sec, latency 3800 usec
```

Результаты для `max_queue_delay_microseconds: 500`. Наблюдаются улучшения значений (`troughput` увеличилась, `latency` уменьшилась).

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1345.5 infer/sec, latency 742 usec
Concurrency: 2, throughput: 1867.87 infer/sec, latency 1069 usec
Concurrency: 3, throughput: 2094.27 infer/sec, latency 1431 usec
Concurrency: 4, throughput: 2023.36 infer/sec, latency 1975 usec
Concurrency: 5, throughput: 2125.16 infer/sec, latency 2351 usec
Concurrency: 6, throughput: 2124.73 infer/sec, latency 2822 usec
Concurrency: 7, throughput: 1675.43 infer/sec, latency 4176 usec
Concurrency: 8, throughput: 2014.29 infer/sec, latency 3970 usec
```

Результаты для `max_queue_delay_microseconds: 50`. Положительная тенденция сохраняется.

```bash
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 1574.56 infer/sec, latency 634 usec
Concurrency: 2, throughput: 2596.02 infer/sec, latency 769 usec
Concurrency: 3, throughput: 2468.67 infer/sec, latency 1214 usec
Concurrency: 4, throughput: 2376.18 infer/sec, latency 1682 usec
Concurrency: 5, throughput: 2297.3 infer/sec, latency 2175 usec
Concurrency: 6, throughput: 2251.21 infer/sec, latency 2663 usec
Concurrency: 7, throughput: 2226.57 infer/sec, latency 3142 usec
Concurrency: 8, throughput: 2296.77 infer/sec, latency 3480 usec
```

Результаты для `max_queue_delay_microseconds: 40`. Присуствует Warnings.

**Выводы**
При `max_queue_delay_microseconds: 50` для *dynamic_batching* были зафиксированы наилучшие значения показателей (`troughput` наибольшее, а `latency` наименьшее).
