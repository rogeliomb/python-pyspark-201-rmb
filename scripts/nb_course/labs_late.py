"""Labs M04–M07: guion paso a paso (tú creas el notebook)."""
from __future__ import annotations

from .common import CELDA_0, comprueba, errores, lab_abre, md, paso, prueba, reto, siguiente


def m04_01() -> list:
    return [
        md(
            lab_abre(
                "M04-01",
                "Joins",
                "M04-01-joins.ipynb",
                "Medir cuántas líneas y pedidos se pierden al hacer inner contra clientes, y listar los huérfanos.",
                "01-teoria.ipynb",
                "03-lab-kpis.ipynb",
            )
        ),
        *paso(
                "1",
                "Carga fact y clientes",
                "El fact ya trae customer_id de la cabecera. Leo Parquet, no CSV.",
                CELDA_0
                + """

spark = get_spark("novashop-m04")
fact = spark.read.parquet(str(STAGING / "fact_lines"))
customers = spark.read.parquet(str(STAGING / "customers_clean"))
print(fact.count(), customers.count())""",
                "`1980 250`.",
                "Si falta fact_lines, cierra M03-02 primero.",
            ),
        *paso(
                "2",
                "Inner frente a left",
                "La diferencia ES el síntoma de las claves huérfanas. Cuento los dos.",
                """inner = fact.join(customers, "customer_id", "inner")
left = fact.join(customers, "customer_id", "left")
print("inner", inner.count(), "left", left.count())""",
                "inner **1956** · left **1980**.",
                "Si el inner sale mayor que 1980, el join de catálogo te ha duplicado (no lo hagas aquí).",
            ),
        *paso(
                "3",
                "Anti-join de huérfanos",
                "left_anti = está en el fact y no en clientes. Mejor que un where a ciegas.",
                """orphans = fact.join(customers, "customer_id", "left_anti")
orphans.select("order_id", "customer_id").distinct().orderBy("order_id").show()
print("líneas", orphans.count(), "pedidos", orphans.select("order_id").distinct().count())""",
                "**24** líneas · **8** pedidos · `customer_id` tipo `CX*`.",
                "Si no ves CX*, los filtraste en M02-03: regenera staging.",
                extra="Opcional: left a `products_clean` por `product_id`. Las líneas `P999` aparecen con `name` nulo: mismo patrón.",
            ),
        md(
            comprueba(
                "`fact.count() - inner.count()` → **24** líneas (8 pedidos). Anótalo en Markdown."
            )
        ),
        *reto(
                "Mismo patrón a grano pedido",
                "Inner/left de `orders_clean` ⋈ `customers_clean`. Markdown que compare con el grano línea.",
                """```python
orders = spark.read.parquet(str(STAGING / "orders_clean"))
print("orders", orders.count())
print("inner", orders.join(customers, "customer_id", "inner").count())  # 780
print("left ", orders.join(customers, "customer_id", "left").count())   # 788
```""",
            ),
        md(
            errores(
                [
                    ("Inner > 1980", "Productos duplicados en otro join", "`dropDuplicates` en product_id"),
                    ("No veo CX*", "Los filtraste en M02-03", "Regenera staging: solo quitas customer_id vacío"),
                    ("customer_id ambiguo", "Join mal nombrado", "Usa `join(..., \"customer_id\")`"),
                ]
            )
        ),
        md(siguiente("03-lab-kpis.ipynb", "M04-02 KPIs")),
    ]


def m04_02() -> list:
    return [
        md(
            lab_abre(
                "M04-02",
                "KPIs",
                "M04-02-kpis.ipynb",
                "Calcular GMV cobrable, nº de pedidos cobrables, ticket medio y tasa de cancelación sobre el universo **con cliente real**.",
                "02-lab-joins.ipynb",
                "04-lab-segmentacion.ipynb",
            )
        ),
        *paso(
                "1",
                "Universo de venta",
                "KPI de dinero ≠ KPI de operativa. Inner a clientes y solo is_billable para el dinero.",
                CELDA_0
                + """

from pyspark.sql.functions import col

spark = get_spark("novashop-m04")
fact = spark.read.parquet(str(STAGING / "fact_lines"))
customers = spark.read.parquet(str(STAGING / "customers_clean"))
sales = (
    fact.join(customers, "customer_id", "inner")
    .where(col("is_billable"))
)
print(sales.count())""",
                "**1122** líneas cobrables con cliente (1127 − 5 paid huérfanas).",
                "Si usas left, atribuyes GMV a CX*.",
            ),
        *paso(
                "2",
                "Cuatro métricas globales",
                "El ticket medio se calcula a grano pedido: sum(GMV) / countDistinct(order_id), no avg de línea.",
                """from pyspark.sql.functions import sum as fsum, countDistinct, round as fround

kpis = sales.agg(
    fround(fsum("gmv_line"), 2).alias("gmv"),
    countDistinct("order_id").alias("orders"),
)
kpis = kpis.withColumn("aov", fround(col("gmv") / col("orders"), 2))
kpis.show()""",
                "GMV ≈ **400157.73** · pedidos cobrables **469** · AOV ≈ **853**.",
                "Si casteaste a double, el céntimo puede moverse: redondea a 2 decimales.",
            ),
        *paso(
                "3",
                "Tasa de cancelación",
                "El denominador es pedidos (no líneas). Sobre orders_clean inner clientes (780).",
                """from pyspark.sql.functions import avg

orders = spark.read.parquet(str(STAGING / "orders_clean"))
ord_ok = orders.join(customers, "customer_id", "inner")
cancel = ord_ok.agg(
    avg((col("status") == "cancelled").cast("double")).alias("cancel_rate")
)
cancel.show()
print("pedidos con cliente", ord_ok.count())""",
                "≈ **0.22**. Pedidos con cliente **780**.",
                "Si mides sobre `sales` (solo paid), la tasa sale 0.",
            ),
        *paso(
                "4",
                "KPI por canal",
                "channel_norm (no channel) evita partir web/WEB. Ordeno por GMV.",
                """(
    sales.groupBy("channel_norm")
    .agg(
        fround(fsum("gmv_line"), 2).alias("gmv"),
        countDistinct("order_id").alias("orders"),
    )
    .orderBy(col("gmv").desc())
    .show()
)""",
                "Cuatro filas (`app`, `other`, `store`, `web`). `web` o `app` en cabeza.",
                "Este groupBy es el cuadro de mando.",
            ),
        md(
            comprueba(
                """Reproduce `gmv / countDistinct(order_id)` solo con is_billable e inner.
Un número ~850, no ~350 (eso sería media de línea). Escríbelo en Markdown."""
            )
        ),
        *reto(
                "GMV por mes y país",
                "`groupBy(\"order_month\", \"country\")` con la misma regla cobrable. `UNK` aparece si no rellenaste país.",
                """```python
(
    sales.groupBy("order_month", "country")
    .agg(fround(fsum("gmv_line"), 2).alias("gmv"))
    .orderBy("order_month", "country")
    .show(20)
)
```""",
            ),
        md(
            errores(
                [
                    ("GMV ~ 2×", "Join al catálogo duplicado", "`dropDuplicates([\"product_id\"])`"),
                    ("AOV ridículamente bajo", "`avg(\"gmv_line\")`", "`sum / countDistinct(order_id)`"),
                    ("Cancel rate 0", "Mediste sobre sales (solo paid)", "Usa orders_clean"),
                ]
            )
        ),
        md(siguiente("04-lab-segmentacion.ipynb", "M04-03 segmentación")),
    ]


def m04_03() -> list:
    return [
        md(
            lab_abre(
                "M04-03",
                "Segmentación",
                "M04-03-segmentacion.ipynb",
                "Clasificar clientes con venta cobrable en low / mid / high según GMV y contar cada banda.",
                "03-lab-kpis.ipynb",
                "../M05-analisis-avanzado/01-teoria.ipynb",
            )
        ),
        *paso(
                "1",
                "GMV por cliente",
                "La segmentación es una agregación DESPUÉS de fijar el grano. Parto de sales cobrable.",
                CELDA_0
                + """

from pyspark.sql.functions import col, sum as fsum, countDistinct, when, lit

spark = get_spark("novashop-m04")
fact = spark.read.parquet(str(STAGING / "fact_lines"))
customers = spark.read.parquet(str(STAGING / "customers_clean"))
sales = fact.join(customers, "customer_id", "inner").where(col("is_billable"))
customer_gmv = sales.groupBy("customer_id", "country", "segment").agg(
    fsum("gmv_line").alias("gmv"),
    countDistinct("order_id").alias("orders"),
)
customer_gmv.orderBy(col("gmv").desc()).show(5)
print(customer_gmv.count())""",
                "~**211** clientes con al menos un paid.",
                "Si agregas *todos* los clientes con left, inflas con GMV nulo.",
            ),
        *paso(
                "2",
                "Bandas de negocio",
                "Umbrales explícitos: <1000 low, <3000 mid, resto high. Encadena when bien (no solapes).",
                """banded = customer_gmv.withColumn(
    "value_band",
    when(col("gmv") < 1000, lit("low"))
    .when(col("gmv") < 3000, lit("mid"))
    .otherwise(lit("high")),
)
banded.groupBy("value_band").count().orderBy("value_band").show()""",
                "`high` ≈ 40 · `low` ≈ 56 · `mid` ≈ 115. Suma = count de customer_gmv.",
                "Los quintiles (`ntile`) van en la mejora, no aquí.",
            ),
        *paso(
                "3",
                "Guarda para M05",
                "M05 rankea sobre este grano sin recalcular el GMV.",
                """banded.write.mode("overwrite").parquet(str(STAGING / "customer_gmv"))
print(spark.read.parquet(str(STAGING / "customer_gmv")).count())""",
                "Carpeta `data/staging/customer_gmv` y el mismo count (~211).",
                "overwrite para poder repetir el lab.",
            ),
        md(
            comprueba(
                "`low + mid + high` debe igualar `customer_gmv.count()`. Una sola cifra, sin clientes en dos bandas."
            )
        ),
        *reto(
                "Quintiles",
                "Usa `ntile(5)` sobre `gmv` (ventana global `orderBy(gmv)`) y cuenta cada quintil. Sin partitionBy: ranking de la compañía.",
                """```python
from pyspark.sql.window import Window
from pyspark.sql.functions import ntile

w = Window.orderBy(col("gmv"))
customer_gmv.withColumn("q", ntile(5).over(w)).groupBy("q").count().orderBy("q").show()
```""",
            ),
        md(
            errores(
                [
                    ("250 clientes en las bandas", "Left con GMV nulo", "Parte de sales cobrable"),
                    ("Un cliente en two bands", "Whens solapados", "`< 1000` luego `< 3000` luego otherwise"),
                ]
            )
        ),
        md(siguiente("../M05-analisis-avanzado/01-teoria.ipynb", "M05 — teoría")),
    ]


def m05_01() -> list:
    return [
        md(
            lab_abre(
                "M05-01",
                "Ranking por ventana",
                "M05-01-ranking-ventana.ipynb",
                """La misma ventana de la teoría, a tamaño NovaShop: top 10 clientes por GMV y, **dentro de cada cliente**, sus 3 productos que más dinero dejan.

No hace falta memorizar la API. En cada paso: ejecuta → mira `rn` → **cambia un número o quita un `partitionBy`** y vuelve a ejecutar. Si solo pegas, no has visto la ventana.""",
                "01-teoria.ipynb",
                "03-lab-acumulados.ipynb",
            )
        ),
        *paso(
                "1",
                "Top 10 de la compañía",
                """Parte de `customer_gmv` (M04-03: una fila por cliente con venta cobrable). Sin `partitionBy`, el ranking es **de toda la empresa**: un solo `rn=1`.

`row_number` pone 1 al GMV más alto (`orderBy desc`), 2 al siguiente, etc. El `where rn <= 10` es el top.""",
                CELDA_0
                + """

from pyspark.sql.functions import col, row_number, sum as fsum
from pyspark.sql.window import Window

spark = get_spark("novashop-m05")
# Una fila = un cliente (sale de M04-03). Si PATH falla: rehaz ese lab o run_pipeline no basta
# (customer_gmv lo escribes tú en M04-03).
cust = spark.read.parquet(str(STAGING / "customer_gmv"))
print("clientes con GMV cobrable", cust.count())

# Sin partitionBy: un único ranking para toda la tabla
w_global = Window.orderBy(col("gmv").desc())
top10 = (
    cust.withColumn("rn", row_number().over(w_global))  # 1 = el que más factura
    .where(col("rn") <= 10)
)
top10.orderBy("rn").show()""",
                "10 filas, `rn` de 1 a 10, GMV hacia abajo. El nº 1 ronda **6000 €**.",
                "Si no tienes `customer_gmv`, no es M05: vuelve a M04-03 (groupBy cliente).",
                if_fail="PATH not found → el Parquet vive en `data/staging/customer_gmv` (lo escribes en el lab de segmentación).",
            ),
        *prueba(
                "Cambia el corte del top",
                "En la celda de arriba, cambia `<= 10` por `<= 3` y vuelve a ejecutar. Luego prueba `<= 1`.",
                """print("filas top3", top10.where(col("rn") <= 3).count())  # 3
# ¿El customer_id del rn=1 sigue siendo el mismo que con top 10?
top10.where(col("rn") == 1).select("customer_id", "gmv").show()""",
                "`<= 3` da 3 filas. El nº 1 **no cambia** (solo recortas). Si cambia, reordenaste mal.",
            ),
        *paso(
                "2",
                "Top 3 productos **por** cliente",
                """Ahora el ranking se **reinicia** en cada persona. Eso es `partitionBy("customer_id")`.

Antes hay que **juntar líneas del mismo producto**: si rankeas el fact a palo seco, la misma SKU sale muchas veces (una por línea). Por eso `groupBy(customer_id, product_id)` y luego la ventana.""",
                """fact = spark.read.parquet(str(STAGING / "fact_lines"))
customers = spark.read.parquet(str(STAGING / "customers_clean"))

# Dinero cobrable de cada par cliente+producto (ya no es grano línea)
product_gmv = (
    fact.join(customers, "customer_id", "inner")
    .where(col("is_billable"))
    .groupBy("customer_id", "product_id")
    .agg(fsum("gmv_line").alias("gmv"))
)

# El rn vuelve a 1 en CADA customer_id
w_prod = Window.partitionBy("customer_id").orderBy(col("gmv").desc())
top3 = (
    product_gmv.withColumn("rn", row_number().over(w_prod))
    .where(col("rn") <= 3)
)

# Mira solo al cliente que era nº 1 de la compañía
top_id = top10.select("customer_id").first()["customer_id"]
print("cliente nº 1 de la compañía:", top_id)
top3.where(col("customer_id") == top_id).orderBy("rn").show()
print("filas top3 (todos los clientes)", top3.count())""",
                "Como mucho 3 filas por cliente; `rn` 1–3. `filas top3` ≤ 211 × 3. El nº 1 de *ese* cliente es un producto, no el ranking global.",
                "Si ves 30 filas del mismo cliente, rankeaste líneas: faltó el groupBy producto.",
            ),
        *prueba(
                "Quita el partitionBy del top 3",
                "Crea `w_mal = Window.orderBy(col(\"gmv\").desc())` (sin partitionBy), calcula `rn` y filtra `rn <= 3`. Compáralo con `top3`.",
                """w_mal = Window.orderBy(col("gmv").desc())  # ranking de TODA la empresa
mal = product_gmv.withColumn("rn", row_number().over(w_mal)).where(col("rn") <= 3)
print("sin partitionBy, filas", mal.count())  # 3 en total, no 3 por cliente
mal.show()
print("con partitionBy, filas", top3.count())""",
                "Sin `partitionBy`: **3 filas en toda la tabla**. Con él: cientos (3 por cliente). Anota los dos counts en Markdown.",
            ),
        md(
            comprueba(
                """Elige un `customer_id` con varios productos y mira sus `rn`.
Empiezan en **1** (no continúan el 1–10 de la compañía). Markdown: id + tres filas.

También: count sin `partitionBy` vs con él (prueba de arriba)."""
            )
        ),
        *reto(
                "rank vs row_number",
                "Sobre `cust`, añade columnas `row_number`, `rank` y `dense_rank` con el mismo `w_global`. Si hay empate de GMV se ve el salto. Markdown: qué salta y qué no.",
                """```python
from pyspark.sql.functions import rank, dense_rank

cmp_ = (
    cust.withColumn("rn", row_number().over(w_global))
    .withColumn("rk", rank().over(w_global))
    .withColumn("dr", dense_rank().over(w_global))
    .orderBy(col("gmv").desc())
)
cmp_.select("customer_id", "gmv", "rn", "rk", "dr").show(15)
```""",
            ),
        md(
            errores(
                [
                    ("Un solo rn=1 en todo el fact", "Olvidaste partitionBy", "Añádelo para “por cliente”"),
                    ("Top 3 con 30 filas del mismo cliente", "Rankeaste líneas", "groupBy cliente+producto antes"),
                    ("Window sin orderBy", "Ranking indefinido", "Siempre ordena la métrica"),
                    ("No está customer_gmv", "Saltaste M04-03", "Ese lab escribe el Parquet"),
                ]
            )
        ),
        md(siguiente("03-lab-acumulados.ipynb", "M05-02 acumulados")),
    ]


def m05_02() -> list:
    return [
        md(
            lab_abre(
                "M05-02",
                "Acumulados por entidad",
                "M05-02-acumulados.ipynb",
                """Numerar los pedidos de cada cliente (`order_n`) y el GMV cobrable **acumulado** en el tiempo (`gmv_running`).

Misma ventana que la teoría: `partitionBy(cliente)` + `orderBy(fecha)`. Si `gmv_running` baja dentro de un cliente, el orden está mal — no lo copies: **compruébalo**.""",
                "02-lab-ranking-ventana.ipynb",
                "../M06-optimizacion-ejecucion/01-teoria.ipynb",
            )
        ),
        *paso(
                "1",
                "Primero: grano pedido (no línea)",
                """Un pedido con 3 productos no es 3 visitas. Si rankeas o acumulas el fact a palo seco, `order_n` cuenta **líneas**.

Por eso agrupas a `order_id`: fecha del pedido = `min(order_ts)`, dinero = `sum(gmv_line)`.""",
                CELDA_0
                + """

from pyspark.sql.functions import col, min as fmin, sum as fsum, row_number

spark = get_spark("novashop-m05")
orders_gmv = (
    spark.read.parquet(str(STAGING / "fact_lines"))
    .join(spark.read.parquet(str(STAGING / "customers_clean")), "customer_id", "inner")
    .where(col("is_billable"))
    .groupBy("customer_id", "order_id")
    .agg(
        fmin("order_ts").alias("order_ts"),  # un instante por ticket
        fsum("gmv_line").alias("gmv"),       # dinero de todas las líneas del ticket
    )
)
print("pedidos cobrables con cliente", orders_gmv.count())""",
                "**469** (el mismo count que el KPI de M04-02). Si salen **1122**, no agregaste a `order_id`: estás en grano línea.",
                "469 tickets ≠ 1122 líneas. El acumulado “por visita” vive en el ticket.",
            ),
        *prueba(
                "¿Qué pasa si no agrupas?",
                "Cuenta el fact cobrable+inner **sin** el `groupBy` de `order_id`. Compáralo con 469.",
                """lineas = (
    spark.read.parquet(str(STAGING / "fact_lines"))
    .join(spark.read.parquet(str(STAGING / "customers_clean")), "customer_id", "inner")
    .where(col("is_billable"))
)
print("líneas", lineas.count(), "pedidos distintos", lineas.select("order_id").distinct().count())""",
                "`líneas` **1122**, `pedidos distintos` **469**. Si usas 1122 como “nº de pedido”, estás inflando visitas.",
            ),
        *paso(
                "2",
                "Número de pedido y acumulado",
                """Una sola window para las dos columnas: el vecindario es el cliente; el eje es el tiempo.

`row_number` → 1.er, 2.º, 3.er ticket de **esa** persona.
`sum(gmv).over(w)` → dinero desde el primer ticket **hasta este** (incluido).""",
                """from pyspark.sql.window import Window

w = Window.partitionBy("customer_id").orderBy("order_ts")
hist = (
    orders_gmv.withColumn("order_n", row_number().over(w))
    .withColumn("gmv_running", fsum("gmv").over(w))
)
hist.orderBy("customer_id", "order_n").show(12)""",
                "`order_n` 1, 2, 3… **por cliente**. `gmv_running` no decrece dentro del mismo `customer_id`.",
                "Si baja, el `orderBy` de la window no es `order_ts` (o está descendente).",
            ),
        *prueba(
                "Un cliente concreto",
                "Elige un `customer_id` que en el `show` tenga `order_n` ≥ 2. Filtra solo ese id y mira si la fila 2 tiene `gmv_running` ≥ fila 1. Anota id y las dos filas en Markdown.",
                """# Cambia el id por uno que hayas visto con varios pedidos
cid = hist.where(col("order_n") >= 2).select("customer_id").first()["customer_id"]
print("cliente", cid)
hist.where(col("customer_id") == cid).orderBy("order_n").show()""",
                "Al menos dos filas. `gmv_running` de `order_n=2` ≥ el de `order_n=1`. Si no, el orden de la ventana está al revés.",
            ),
        *paso(
                "3",
                "Primera compra vs repetición",
                """`order_n == 1` es la definición de “nuevo” en este curso: primer ticket cobrable de ese cliente. El resto son repeticiones.""",
                """hist.groupBy((col("order_n") == 1).alias("is_first")).count().show()""",
                "~211 primeras compras (`true`: un cliente con paid) y el resto `false` (repeticiones). 211 + repeticiones = 469.",
                "Sin `partitionBy`, `order_n=1` sería **una sola fila en toda la empresa**.",
            ),
        *prueba(
                "Ventana de toda la empresa",
                "Repite el paso 2 con `w_emp = Window.orderBy(\"order_ts\")` (sin partitionBy). Cuenta cuántos `order_n == 1` hay.",
                """w_emp = Window.orderBy("order_ts")  # un solo ranking temporal global
hist_emp = orders_gmv.withColumn("order_n", row_number().over(w_emp))
print("order_n=1 sin partitionBy", hist_emp.where(col("order_n") == 1).count())  # 1
print("order_n=1 con partitionBy", hist.where(col("order_n") == 1).count())     # ~211""",
                "Sin `partitionBy`: **1**. Con él: ~**211**. Esa diferencia *es* la ventana.",
            ),
        md(
            comprueba(
                """Un cliente con `order_n` ≥ 2: `gmv_running` fila 2 ≥ fila 1. Markdown con el id.

Counts: 469 pedidos; ~211 primeros; `order_n=1` global (sin partitionBy) = 1."""
            )
        ),
        *reto(
                "Pedidos hasta superar 1000 €",
                "Quédate, por cliente, con la **primera** fila donde `gmv_running >= 1000` (o ninguna si no llega). Markdown: ¿`order_n` 1 o hace falta el 2.º ticket?",
                """```python
w2 = Window.partitionBy("customer_id").orderBy("order_ts")
crossed = hist.where(col("gmv_running") >= 1000)
first_cross = crossed.withColumn("rn", row_number().over(w2)).where(col("rn") == 1)
first_cross.select("customer_id", "order_n", "gmv_running").show()
print("clientes que cruzan 1000", first_cross.count())
```""",
            ),
        md(
            errores(
                [
                    ("gmv_running igual en todas las filas", "Window sin orderBy", "partitionBy + orderBy(order_ts)"),
                    ("1122 “pedidos”", "No agregaste a order_id", "Paso 1"),
                    ("Acumulado a nivel empresa", "Falta partitionBy", "Añádelo"),
                    ("order_n=1 solo una vez", "Ventana global", "partitionBy(customer_id)"),
                ]
            )
        ),
        md(siguiente("../M06-optimizacion-ejecucion/01-teoria.ipynb", "M06 — teoría")),
    ]

def m06_01() -> list:
    return [
        md(
            lab_abre(
                "M06-01",
                "Explain y DAG",
                "M06-01-explain-dag.ipynb",
                """Dejar de pensar como Pandas: escribir `where` **no** recorre el fichero. Spark solo alarga una **receta**. `explain` es ver qué haría **sin** hacerlo. `count` es hacerlo (ahí nace un job en la UI).

Orden de este lab (si lo inviertes, no se nota nada): **print → explain → count**. Primero el mapa, luego el viaje.""",
                "01-teoria.ipynb",
                "03-lab-cache-particionado.ipynb",
            )
        ),
        md(
            """## Qué queremos conseguir (léelo; el código viene después)

Vienes de Pandas (o de “ejecuto la celda y ya está filtrado”). En Spark **no**.

Hasta que no lanzas una **acción** (`count`, `show`, `write`), Spark **no** ha leído las 1980 líneas de `fact_lines`. Los `where` / `select` solo alargan una receta. Por eso `print(planned)` no te da un número: te enseña el **objeto** (el papel de la receta).

`explain` es la herramienta de este lab: **ver qué haría Spark sin recorrerlo**. El muro de texto no se lee como un libro; se **busca** Scan (de dónde salen las filas) y Filter (tus `where`).

`count` es el viaje: ahora sí abre el Parquet, aplica la receta y te da un entero. En Spark UI aparece **un job nuevo**. El dibujo de ese job (etapas) y el árbol de la pestaña SQL son *la misma historia* que el `explain`, pero **después** de haber ejecutado.

Eso es todo. No memorices Catalyst. El lab siguiente (cache) solo tiene sentido si esto está claro: “esta receta es cara → la guardo”.

| Escribes | ¿Recorre `fact_lines`? | Qué ves en el notebook | Qué ves en la UI |
|----------|------------------------|------------------------|------------------|
| `where` / `select` / `read` | no | nada (o el objeto si haces `print`) | Jobs **igual** |
| `print(df)` | no | `DataFrame[columnas…]`, **sin** entero | Jobs **igual** |
| `df.explain(...)` | no | muro de texto (receta) | Jobs **igual** |
| `df.count()` / `show()` | **sí** | un entero / filas | **un job nuevo** |

El número de filas del `count` **no** es la demo (ya sabías filtrar en M03). La demo es: Job Id **quieto** con print/explain, Job Id **+1** con count.

### Markdown de *tu* notebook: no improvises teoría

El plantilla de los pasos dice “con tus palabras”. **Aquí no.** En cada paso te dejo el párrafo ya redactado, con huecos `___` (Job Ids, sí/no). Cópialo, ejecuta, rellena. Si escribes otra teoría, este lab vuelve a “no demostrar nada”.

Necesitas `data/staging/fact_lines` (M03-02 o `python3 scripts/run_pipeline.py`)."""
        ),
        md(
            """## Spark UI: ábrela **antes** del paso 1 (una sola vez)

Sin esto, cada paso te va a pedir “mira Jobs” y no vas a saber a qué ventana te refieres.

1. En el Codespace / VS Code, panel de abajo: pestaña **Ports** (junto a Terminal). Si no está: *View → Ports*.
2. Busca el puerto **4040**. Suele poner `Spark` o el nombre de la app.
3. Clic en el **globo** (Open in Browser) o *Forward*. Se abre una web “Spark … Application UI”.
4. Arriba, el nombre de la app debe ser `novashop-m06` **después** de ejecutar `get_spark` del paso 1. Si aún no lo has ejecutado, abre la UI igual y **recarga** tras el paso 1.
5. Si **4040** no carga o está vacío: mira si hay **4041**. Otra sesión se quedó el 4040. En una celda: `spark.stop()`, otra vez `get_spark("novashop-m06")`, recarga Ports y entra al puerto que haya salido.
6. Pestañas que usamos en **este** lab (las otras, ignóralas):

| Pestaña | Qué es, en cristiano | Cuándo hay algo que mirar |
|---------|----------------------|---------------------------|
| **Jobs** | Lista de **viajes**. Cada `count`/`show`/`write` = una fila nueva. | Solo **después** de una acción. Print y explain **no** añaden fila. |
| **SQL / DataFrame** | El **árbol** de esa consulta (Scan, Filter, Aggregate…). Es el `explain` después de haber viajado. | Después del primer `count`. |
| **Storage** | Datasets guardados en memoria (`cache`). | En este lab debe quedarse **vacío**. Si no, kernel sucio: `spark.catalog.clearCache()`. |

**Cómo se lee Jobs** (la tabla, no el dibujo):

- **Job Id** — 0, 1, 2… El más alto es el último viaje. Si ya corriste la teoría en este kernel, **no empieza en 0**. Da igual: lo que importa es si **sube** o no.
- **Description** — casi siempre se ve `count` (o `show`). Es el nombre de la acción, no de tu DataFrame.
- **Duration** — en local puede ser 0,1 s. No es la prueba.
- **Stages** — `2/2` o `1/1`. Un job se parte en **etapas** cuando hay shuffle. Un `count` típico: etapa que lee+filtra+cuenta trozos, a veces otra que junta el total (`Exchange` por medio).

**El dibujo (DAG) no es el `explain`.**

- En **Jobs**, al cliclar un Job Id, el “DAG Visualization” son **cajas grandes = stages**. Vas a ver nombres tipo *WholeStageCodegen*, *Exchange*, *count*. **No busques la palabra `Filter` ahí.** El filtro va *dentro* de un stage y esa caja no lo detalla.
- El árbol con *Scan parquet* y *Filter* está en **SQL / DataFrame** → clic en la consulta → Details. Eso **sí** se parece al `explain` del notebook.
- Si abres un DAG y el Job Id es **menor o igual** al que anotaste *antes* de la celda, es un viaje **viejo**. No es de esta celda.

**Recarga.** La UI no siempre pinta sola: vuelve a cliclar **Jobs** (o F5) después de cada celda.

Anota **ahora**, antes de pegar código, el Job Id más alto que ves (si no hay ninguno, escribe `ninguno / -1`):

`Job Id de partida = ___`"""
        ),
        *paso(
                "1",
                "Encadenar no ejecuta (solo alarga la receta)",
                """Ignora lo de “con tus palabras”. Copia esto en el Markdown y rellena **después** de ejecutar:

```
Paso 1. Escribir where no recorre el Parquet: solo alarga la receta.
print(planned) me enseñó el objeto DataFrame[…], no un entero.
Job Id de partida: ___
Job Id después de esta celda: ___   (tiene que ser el mismo)
En Jobs no hay fila nueva. Storage sigue vacío. No hay DAG nuevo que abrir.
```

**UI de este paso (hazlo en este orden):**

1. Mira Jobs. Anota el Job Id más alto (el de partida, si aún no lo tenías).
2. Ejecuta la celda de código (`Shift+Enter`).
3. Vuelve a la UI → **Jobs** (recarga). El más alto **no** cambia.
4. Abre **Storage**. Vacío.
5. **No** clicles un job viejo para “ver el DAG”. Ese DAG no es de esta celda. Esta celda **no** ha viajado.

Si el Job Id **sube** aquí: tienes un `count`/`show` de más en la celda, o `get_spark` disparó algo raro. Quita cualquier `.count()` y reejecuta.""",
                CELDA_0
                + """

from pyspark.sql.functions import col

spark = get_spark("novashop-m06")

# Tres where + select = receta más larga. Todavía NO hay viaje.
planned = (
    spark.read.parquet(str(STAGING / "fact_lines"))
    .where(col("is_billable"))
    .where(col("gmv_line") > 0)
    .where(col("channel_norm").isin("web", "app"))
    .select("order_id", "customer_id", "gmv_line", "order_month")
)

# Objeto (papel de la receta), NO el número de filas
print("¿qué es?", type(planned).__name__)
print(planned)""",
                "`DataFrame[order_id, customer_id, gmv_line, order_month]`. Ningún entero tipo 800. Job Id **igual** que el de partida.",
                "Si `print` ya te diera 1127, estarías ejecutando un `count` sin darte cuenta. Pandas ya habría filtrado; Spark no.",
                if_fail="PATH / AnalysisException de `fact_lines` → falta M03-02 o `run_pipeline.py`.",
            ),
        *paso(
                "2",
                "`explain`: ver qué haría **sin** hacerlo",
                """Copia y rellena:

```
Paso 2. explain imprime la receta. No recorre el Parquet. No da el count.
Job Id antes: ___   Job Id después: ___   (igual)
En el helper: fact_lines sí, is_billable sí, Filter sí.
En el tocho, Ctrl+F: una línea de Scan/fact_lines y una de Filter/PushedFilters.
Jobs: sin fila nueva. El árbol Scan+Filter AÚN no está en SQL: no hemos viajado.
```

**Cómo leer el tocho** (no de arriba abajo): Ctrl+F en la salida de la celda.

| Buscas | Quiere decir |
|--------|----------------|
| `fact_lines` / *FileScan* / *Scan parquet* | de aquí salen las filas |
| `is_billable` / `gmv_line` / `channel_norm` | tus `where` están en la receta |
| *PushedFilters* | el `where` se empujó al scan (no “leer todo y filtrar al final del todo”) |
| *Filter* | un `where` como paso |
| *Project* | el `select` de cuatro columnas |

El helper de la celda te imprime **sí/NO** para no perderte. El tocho es por si quieres copiar **una** línea de Scan y **una** de Filter.

**UI de este paso:**

1. Anota Job Id más alto.
2. Ejecuta. El notebook se llena de texto; **no** sale el entero del count.
3. Recarga **Jobs**: el más alto **sigue igual**. `explain` no es un viaje.
4. **SQL / DataFrame**: igual que antes (vacío, o las consultas *viejas*). Esta receta todavía no se ejecutó, así que **no** aparece un árbol nuevo.
5. **Storage**: vacío.

Esto es el aha: ya viste Scan y Filter **sin** haber leído el Parquet.""",
                """def receta(df, etiqueta):
    # Versión corta del explain: sí/NO, sin leer el tocho como un libro
    txt = df._jdf.queryExecution().executedPlan().toString()
    print("=== Receta de", etiqueta, "(aún NO ha corrido) ===")
    for pal in (
        "fact_lines",
        "is_billable",
        "gmv_line",
        "channel_norm",
        "Filter",
        "Project",
    ):
        print("   ", "sí" if pal in txt else "NO", pal)

receta(planned, "planned")
print("----- tocho (Ctrl+F: Scan y Filter) -----")
planned.explain("formatted")""",
                "Helper: `sí` en `fact_lines`, `is_billable`, `Filter`. El tocho no termina en un entero. Job Id **igual**.",
                "Si Job Id sube, no has hecho `explain`: has hecho `count`. Si el helper dice NO en `is_billable`, estás explain-ando otro DataFrame.",
            ),
        *paso(
                "3",
                "`count`: ahora sí viaja (un job, entonces hay qué pintar)",
                """Copia y rellena:

```
Paso 3. count recorre el Parquet y aplica la receta que ya vimos en el explain.
Filas = ___   (varios cientos; cobrable + GMV>0 + web/app)
Job Id antes: ___   Job Id después: ___   (este SÍ sube: +1)
En Jobs: una fila nueva, Description con "count".
En SQL: una consulta nueva; al abrirla veo Scan y Filter (el explain, pero ejecutado).
En el DAG de Jobs (cajas grandes) NO busqué la palabra Filter; son stages.
Storage sigue vacío (no hay cache).
```

**UI de este paso (aquí por fin hay dibujo):**

1. Anota Job Id más alto (el de los pasos 1–2).
2. Ejecuta la celda. En el notebook: un **entero**.
3. Recarga **Jobs**. Hay **una fila nueva**. El más alto es el anterior **+1**. Description: `count` (el sitio del `NativeMethod…` da igual). Duration: irrelevante. Stages: `1/1` o `2/2` (el 2.º junta los recuentos parciales).
4. Clic en **ese** Job Id (el nuevo, no uno viejo).
   - “DAG Visualization”: cajas azules = **stages**, no cada operador. Nombres tipo *WholeStageCodegen*, *Exchange*, *count*. **No está mal que no ponga Filter.** Filter vive *dentro* de la etapa que lee.
   - Si hay **dos** cajas unidas: una leyó y contó trozos; la otra sumó el total. La flecha / *Exchange* es el shuffle de ese count, no un join tuyo.
5. Pestaña **SQL / DataFrame**. Una consulta nueva, misma duración grosera, asociada a ese job. Clic → Details.
   - Aquí sí: *FileScan parquet* / *Scan*, *Filter* / *PushedFilters*, *Project*, y **HashAggregate** (porque `count` suma).
   - Compáralo con el `explain` del paso 2: Scan y Filter son la misma receta. El Aggregate **extra** es el count (el explain de `planned` aún no sumaba: solo iba a producir filas).
6. **Storage**: sigue vacío.

Si cada celda te crea un job, tienes un `.show()` de debug: coméntalo.""",
                """# Viaje: abre fact_lines, aplica la receta, devuelve un entero
n = planned.count()
print("líneas cobrables web/app con GMV > 0 =", n)
print("mira Jobs: una fila nueva. SQL: árbol Scan+Filter+Aggregate")""",
                "Un entero **> 0**, varios cientos. Job Id **+1**. SQL muestra Scan y Filter. Storage vacío.",
                "El entero no es la prueba. La prueba es la fila nueva en Jobs y que el árbol SQL cuadra con el explain del paso 2.",
            ),
        *prueba(
                "Un `where` extra alarga la receta, no viaja",
                """Sigue **sin** `count`. Copia:

```
Prueba. mas_estrecho = planned con gmv_line > 50.
print me dio otro DataFrame[…], no un entero.
Helper: planned no tiene por qué mencionar 50; mas_estrecho sí (gmv_line).
Job Id igual que al terminar el paso 3 (___). No hay fila nueva.
Si contara mas_estrecho, el n sería MENOR y nacería otro job: eso ya sería otro viaje.
```

**UI:** recarga Jobs **antes** y **después**. El más alto no cambia. SQL no añade consulta. El helper es el A/B: dos recetas, cero viajes.""",
                """mas_estrecho = planned.where(col("gmv_line") > 50)
print("sigue siendo un objeto:", mas_estrecho)
receta(planned, "planned (3 where)")
receta(mas_estrecho, "mas_estrecho (4 where; GMV > 50)")
print("Job Id: el de después del count. Si sube, has lanzado un count sin querer.")""",
                "Dos bloques `sí/NO`. `mas_estrecho` sigue mostrando `gmv_line`. Job Id **igual** que al final del paso 3.",
            ),
        md(
            comprueba(
                """Cierra con **este** bloque en Markdown (rellenado; no otro ensayo):

```
Escribir where no recorre el Parquet; alarga la receta.
print enseña el objeto. explain enseña Scan/Filter sin viajar (Job Id ___ → ___ , igual).
count sí viaja: filas = ___ ; Job Id ___ → ___ (+1).
En Jobs el DAG son stages (no busqué Filter). En SQL vi Scan y Filter como en el explain.
Storage vacío en todo el lab.
Un where extra (gmv > 50) no creó job.
```"""
            )
        ),
        *reto(
                "formatted vs el tocho extended (y dónde está en la UI)",
                """`explain("formatted")` es el físico, el de clase. `explain(True)` suelta cuatro capas (parsed → analyzed → optimized → physical): más ruido. Quédate con **Physical Plan** / formatted.

`PushedFilters` junto al FileScan = el `where` se empujó al scan.

En la UI: SQL → tu `count` → Details. Ese árbol es el físico **después** de viajar. El `explain` del paso 2 era el mismo tipo de árbol **antes** de viajar.""",
                """print("===== formatted (el de clase) =====")
planned.explain("formatted")
print("===== extended (busca Physical Plan y PushedFilters; ignora el resto) =====")
planned.explain(True)""",
            ),
        md(
            errores(
                [
                    ("UI vacía / 404", "4040 no reenviado, o la sesión está en 4041", "Ports → globo 4040; o `spark.stop()` + `get_spark()` y prueba 4041"),
                    ("Job Id sube en print/explain", "`count`/`show` en la misma celda, o clicaste un job viejo y pensaste que era nuevo", "Quita acciones; compara el número *antes/después*, no el dibujo viejo"),
                    ("En el DAG de Jobs no veo Filter", "Normal: esas cajas son stages", "Árbol con Filter: pestaña SQL, o el `explain` del notebook"),
                    ("SQL vacío después del count", "No recargaste, o miras otra app (nombre ≠ novashop-m06)", "F5; comprueba el nombre arriba"),
                    ("Storage con algo", "Cache de otra celda / de M06-02", "`spark.catalog.clearCache()`"),
                    ("Cada celda crea un job", "`show` de debug", "Coméntalo mientras mides Jobs"),
                    ("Dos sesiones", "`SparkSession()` a mano", "Solo `get_spark()`"),
                    ("explain → AnalysisException", "Columna mal escrita", "Para planificar también resuelve nombres: corrige el typo"),
                    ("No está fact_lines", "Saltaste M03", "`run_pipeline.py` o lab M03-02"),
                ]
            )
        ),
        md(siguiente("03-lab-cache-particionado.ipynb", "M06-02 cache")),
    ]


def m06_02() -> list:
    return [
        md(
            lab_abre(
                "M06-02",
                "Cache y particionado",
                "M06-02-cache-particionado.ipynb",
                """Dos ideas **distintas** (no las mezcles):

1. **Cache** — “guarda este resultado en memoria para no repetir el plan”. Se llena con una **acción**, no al escribir `.cache()`. La prueba **no** es que el `count` salga 9016 dos veces (eso es normal). La prueba es Storage **0 → 0 → 1 → 1 → 0**.
2. **`repartition`** — baraja filas en **memoria** a N trozos. No crea carpetas en disco (eso es `write.partitionBy` en M07) ni es el `partitionBy` de las ventanas (M05).""",
                "02-lab-explain-dag.ipynb",
                "../M07-persistencia-datos/01-teoria.ipynb",
            )
        ),
        md(
            """## Léelo antes (si no, “no demuestra nada”)

### Palabras que se parecen y no son lo mismo

| Lo que escribes | Dónde vive | Para qué |
|-----------------|------------|----------|
| `Window.partitionBy("customer_id")` (M05) | receta de la ventana | recortar el **vecindario** (por cliente) |
| `df.repartition(12, col("order_month"))` | **memoria**, ahora | barajar a 12 trozos (shuffle) |
| `write.partitionBy("order_month")` (M07) | **disco**, carpetas | `order_month=2024-01/` |
| `df.cache()` | memoria, **después** de una acción | no repetir un plan caro |

### Cómo se demuestra el cache (no mires el 9016)

El número de filas **tiene** que ser igual con cache y sin él. Cache no cambia el resultado; cambia **de dónde** sale la segunda lectura.

| Momento | Storage (pestaña o `n_en_storage()`) | Qué significa |
|---------|--------------------------------------|---------------|
| DataFrame creado | **0** | no hay nada guardado |
| Acabas de escribir `.cache()` | **0** | solo **marcó**; aún no calculó |
| 1.er `count` | **1** | esa acción **llenó** memoria |
| 2.º `count` | **1** | reutiliza; no soltó |
| `unpersist()` | **0** | suelta (si no, el Codespace se llena) |

Los **segundos** del cronómetro en local a veces no se inmutan. Si el reloj miente, Storage no.

Spark UI: Ports **4040** → pestaña **Storage**. Jobs: el 1.er count hace Scan+Filter; el 2.º debería verse más corto / *InMemory*.

Necesitas `fact_lines` (M03). No subas de 8 copias: OOM.""",
        ),
        *paso(
                "1",
                "El fact de siempre + qué es una partición",
                """Una **partición** (aquí) es un **trozo de trabajo en memoria**, no una carpeta. `getNumPartitions()` dice en cuántos trozos está **ahora** este DataFrame.

Leemos las 1980 líneas. Aún no hay cache.""",
                CELDA_0
                + """

from pyspark.sql.functions import col, lit
from time import perf_counter

spark = get_spark("novashop-m06")
spark.catalog.clearCache()  # por si una celda anterior dejó basura

def n_en_storage():
    # Cuántos RDD hay en memoria (= pestaña Storage). 0 vacío, ≥1 hay cache lleno.
    return int(spark.sparkContext._jsc.getPersistentRDDs().size())

base = spark.read.parquet(str(STAGING / "fact_lines"))
print("filas fact_lines", base.count())                 # 1980
print("particiones ahora", base.rdd.getNumPartitions())  # > 1 en local[*]
print("Storage (tiene que ser 0)", n_en_storage())""",
                "`filas fact_lines 1980`. Particiones **> 1**. Storage **0**.",
                "Si sale PATH error, no es este lab: falta M03-02 / pipeline.",
                if_fail="1980 no sale → regenera staging. Storage ≠ 0 → `spark.catalog.clearCache()` y reejecuta.",
            ),
        *paso(
                "2",
                "Por qué apilamos 8 copias (lupa, no producción)",
                """1980 filas (como las 20 de la teoría) son tan pocas que el 1.er y el 2.º `count` duran igual y parece que el cache “no hace nada”.

`unionByName` **apila** la misma tabla 8 veces: 1980 × 8 = **15840** filas. `_copy` marca de qué copia sale cada fila. No es un patrón de pipeline; es para que Storage tenga algo que mostrar y el job dure un poco.

`range(7)` + la base con `_copy = -1` → 8 copias en total.""",
                """xl = base.withColumn("_copy", lit(-1))
for i in range(7):
    xl = xl.unionByName(base.withColumn("_copy", lit(i)))

print("filas apiladas", xl.count())                       # 15840
print("particiones tras el union", xl.rdd.getNumPartitions())
print("Storage sigue", n_en_storage())                    # 0: aún no hay cache""",
                "**15840**. Storage **0**. Varias particiones.",
                "Si cuentas 1980, el bucle no se ejecutó (reusas `base` en vez de `xl`).",
            ),
        *paso(
                "3",
                "Dos counts **fríos** (sin cache): el plan se repite",
                """Sin cache, cada `count` vuelve a leer Parquet + unions + filtro cobrable.

`timed_count` mide reloj **y** imprime filas. Los dos tiempos son del **mismo orden**. El número **9016** las dos veces no demuestra cache: demuestra que el filtro es el mismo (1127 cobrables × 8).

Storage sigue en 0: no hemos marcado nada.""",
                """def timed_count(df, label):
    t0 = perf_counter()
    n = df.count()
    print(label, "n =", n, "s =", round(perf_counter() - t0, 3),
          "Storage =", n_en_storage())
    return n

# Filtro cobrable AQUÍ (no dentro de timed_count: si no, no se entiende qué cacheas)
billable = xl.where(col("is_billable"))

timed_count(billable, "1er count frío")
timed_count(billable, "2º count frío")""",
                "Las dos veces **n = 9016**. Storage **0** y **0**. Tiempos parecidos (si no, mira Jobs: son dos jobs del mismo estilo, Scan otra vez).",
                "Frío = cada acción rehace el plan. Aún no hay atajo.",
            ),
        *paso(
                "4",
                "`.cache()` **solo marca** — Storage sigue vacío",
                """`cache()` quiere decir: “la **próxima** acción, guarda el resultado”. Hasta que no haya `count`/`show`, Storage = 0.

Ejecuta **esta** celda y **párate**. Mira UI → Storage (vacío) y el print. No lances el count todavía.""",
                """warm = billable.cache()
print("acabo de escribir cache(); Storage =", n_en_storage())  # 0
print("storageLevel (la intención, no los datos):", warm.storageLevel)
print("PARA AQUÍ. Storage tiene que seguir vacío.")""",
                "Storage **0**. `storageLevel` ya dice MEMORY (eso es la marca, no el llenado).",
                "Si aquí ya ves 1, el kernel tenía un cache viejo: `clearCache()` y desde el paso 1.",
            ),
        *paso(
                "5",
                "El 1.er count llena; el 2.º reutiliza",
                """Ahora sí: la primera acción **materializa**. La segunda debería leer memoria.

El 9016 **otra vez** es correcto: cache no cambia el resultado. Lo que cambia es Storage 0→**1** y, en `explain`, *InMemoryTableScan* / *InMemoryRelation*.

No hagas `unpersist` en esta celda: si no, al terminar Storage vuelve a 0 y “no se ve nada”.""",
                """timed_count(warm, "1º count (LLENA el cache)")
print("   → Storage tiene que ser 1. Mira también UI → Storage.")

timed_count(warm, "2º count (lee cache, mismo n)")
print("   → Storage sigue 1 (no lo soltó)")

print("--- Plan del 2º count: busca InMemoryTableScan / InMemoryRelation ---")
warm.explain("formatted")""",
                "1º: n=9016, Storage **1**. 2º: n=9016, Storage **1**. En el explain aparece *InMemory*. El reloj del 2º no empeora (a veces es igual de corto: fíate de Storage).",
                "Dos 9016 no son la demo. Un Storage que pasa a 1 sí.",
            ),
        *paso(
                "6",
                "`unpersist` suelta (otra celda, siempre)",
                """Si no sueltas, el Codespace se queda con el fact×8 en memoria. `unpersist()` vacía Storage.

Jupyter a veces pinta `DataFrame[...]` porque `unpersist` **devuelve** el DataFrame: ignóralo, o deja el `print` al final.""",
                """warm.unpersist()
print("después de unpersist, Storage =", n_en_storage())  # 0
print("listo")""",
                "Storage **0**. UI → Storage vacío otra vez.",
                "Si vuelves a `warm.count()` ahora, **recalcula** (frío) y, como ya no está cacheado, Storage sigue 0.",
            ),
        *prueba(
                "Cache sin acción = Storage vacío",
                "Sin pegar el paso 5: `otra = billable.cache()` y `print(n_en_storage())`. Luego un `count` y otra vez el print. `unpersist` al terminar.",
                """otra = billable.cache()
print("solo cache(), Storage =", n_en_storage())  # 0
_ = otra.count()
print("tras count, Storage =", n_en_storage())    # ≥ 1
otra.unpersist()
print("tras unpersist, Storage =", n_en_storage())  # 0""",
                "**0**, luego **≥ 1**, luego **0**. Escríbelo en Markdown. Eso es todo el lab de cache.",
            ),
        md(
            """## Parte B — `repartition` (esto **no** es cache)

Ya sabes llenar y soltar memoria. Ahora: **cómo está troceado** el DataFrame en RAM.

`repartition(12, col("order_month"))` **baraja** (shuffle) hacia 12 particiones, una intención por mes. Sirve de cara a **escribir** en M07. No crea las carpetas todavía.

Si omites el `12` y pones solo la columna, Spark 3.5 usa **200** particiones por defecto: inútil aquí."""
        ),
        *paso(
                "7",
                "12 particiones por mes (shuffle en memoria)",
                """Vuelve a cachear `billable` un momento (el paso 6 lo soltó) o reparte sobre `billable` directo.

`getNumPartitions() == 12`. El `groupBy("order_month")` debe listar **12** meses de 2024.

Esto **no** es `Window.partitionBy`. Esto **no** es `write.partitionBy`.""",
                """# El cache del paso 5 ya se soltó; no hace falta para contar particiones
by_month = billable.repartition(12, col("order_month"))
print("particiones", by_month.rdd.getNumPartitions())  # 12

(
    by_month.groupBy("order_month")
    .count()
    .orderBy("order_month")
    .show()
)""",
                "`particiones 12`. Doce filas de meses `2024-01` … `2024-12`.",
                "Si ves 200, llamaste `repartition(col(\"order_month\"))` **sin** el 12.",
            ),
        md(
            comprueba(
                """Markdown con la tabla Storage de tu ejecución: 0 (tras cache sin count), 1 (tras 1.er count), 1 (tras 2.º), 0 (unpersist).

Una frase: el 9016 igual las dos veces **no** es la prueba.

`getNumPartitions()` tras el paso 7 = **12**."""
            )
        ),
        *reto(
                "coalesce(1) vs repartition(1)",
                """Los dos dejan 1 partición, pero no igual de caro:

- `repartition(1)` **siempre** shufflea (`Exchange` en el plan).
- `coalesce(1)` **reduce** trozos sin un shuffle amplio.

Útil para un único fichero de entrega; mal hábito si lo pones en mitad del pipeline (M07). Ejecuta los dos `explain` y copia la línea donde uno dice *Exchange* y el otro no.""",
                """print("===== coalesce(1) (sin shuffle amplio) =====")
billable.coalesce(1).explain("formatted")
print("===== repartition(1) (shuffle: busca Exchange) =====")
billable.repartition(1).explain("formatted")
print("particiones coalesce", billable.coalesce(1).rdd.getNumPartitions())
print("particiones repartition", billable.repartition(1).rdd.getNumPartitions())""",
            ),
        md(
            errores(
                [
                    ("Los dos count salen 9016 y “no demuestra nada”", "El número tiene que ser igual", "Mira Storage 0→1→1→0, no el 9016"),
                    ("Storage vacío tras cache()", "No hubo acción, o unpersist en la misma celda", "count en celda aparte; unpersist después"),
                    ("Storage = 1 al escribir cache()", "Cache viejo en el kernel", "`spark.catalog.clearCache()`"),
                    ("OOM / el kernel muere", "Demasiadas copias + cache", "Quédate en 8; `unpersist`"),
                    ("1980 o 200 particiones en el paso 7", "`repartition` sin `12`", "`repartition(12, col(\"order_month\"))`"),
                    ("UI en 4041 / vacía", "Otra SparkSession ocupó 4040", "`spark.stop()`; Ports 4040"),
                ]
            )
        ),
        md(siguiente("../M07-persistencia-datos/01-teoria.ipynb", "M07 — teoría")),
    ]


def m07_01() -> list:
    return [
        md(
            lab_abre(
                "M07-01",
                "Parquet y layout analítico",
                "M07-01-parquet-layout.ipynb",
                "Publicar `data/curated/sales_analytics` en Parquet particionado por mes y demostrar que un filtro de mes no lee el año entero.",
                "01-teoria.ipynb",
                "../../README.md",
            )
        ),
        *paso(
                "1",
                "Dataset curated",
                "Left al catálogo conserva P999. Inner a clientes quita CX*. Solo paid. dropDuplicates en product_id.",
                CELDA_0
                + """

from pyspark.sql.functions import col

spark = get_spark("novashop-m07")
fact = spark.read.parquet(str(STAGING / "fact_lines"))
customers = spark.read.parquet(str(STAGING / "customers_clean"))
products = (
    spark.read.parquet(str(STAGING / "products_clean"))
    .dropDuplicates(["product_id"])
)
sales = (
    fact.join(customers, "customer_id", "inner")
    .join(products, "product_id", "left")
    .where(col("is_billable"))
    .select(
        "order_id", "order_ts", "order_month",
        "customer_id", "country", "segment",
        "product_id", "category",
        "qty", "unit_price", "discount", "gmv_line", "channel_norm",
    )
)
print(sales.count())""",
                "**1122** filas (mismo universo que M04-02).",
                "Si sale 2244, el catálogo no era único.",
            ),
        *paso(
                "2",
                "Escribe Parquet por mes",
                "overwrite deja el curated idempotente. Doce particiones = doce meses de 2024.",
                """dest = CURATED / "sales_analytics"
CURATED.mkdir(parents=True, exist_ok=True)
(
    sales.write.mode("overwrite")
    .partitionBy("order_month")
    .parquet(str(dest))
)
print(sorted(p.name for p in dest.iterdir() if p.is_dir()))""",
                "Carpetas `order_month=2024-01` … `order_month=2024-12` (más `_SUCCESS`).",
                "Esto es layout de disco, no el repartition de M06.",
            ),
        *paso(
                "3",
                "Prune al leer un mes",
                "El plan debe listar solo marzo (o PartitionFilters: order_month=2024-03).",
                """marzo = spark.read.parquet(str(dest)).where(col("order_month") == "2024-03")
marzo.explain("formatted")
print("marzo", marzo.count(), "total", spark.read.parquet(str(dest)).count())""",
                "Total **1122**. `marzo` es un subconjunto. El formatted menciona `2024-03`.",
                "Copia en Markdown la línea del PartitionFilters.",
            ),
        *paso(
                "4",
                "CSV vs Parquet (schema, no solo tamaño)",
                "coalesce(1) solo existe aquí para comparar *un* CSV, no como patrón. Releo los dos schemas.",
                """import os

csv_dir = CURATED / "_csv_compare"
sales.coalesce(1).write.mode("overwrite").option("header", True).csv(str(csv_dir))

def du(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

print("parquet", du(dest), "csv", du(csv_dir))
spark.read.parquet(str(dest)).printSchema()
spark.read.option("header", True).csv(str(csv_dir)).printSchema()""",
                "Parquet mantiene `decimal`/`timestamp`. El CSV vuelve a string. El tamaño: Parquet suele ganar; en este volumen a veces es parecido.",
                "Curated en CSV “para el analista” pierde tipos.",
            ),
        md(
            comprueba(
                """Vuelve a ejecutar el `write.mode(\"overwrite\")` y cuenta.
Sigue **1122**. No se duplica. Anótalo."""
            )
        ),
        *reto(
                "Dos claves de partición",
                "Copia `sales_analytics_geo` con `partitionBy(\"order_month\", \"country\")` y lee marzo ∧ ES. No particiones por customer_id.",
                """```python
geo = CURATED / "sales_analytics_geo"
sales.write.mode("overwrite").partitionBy("order_month", "country").parquet(str(geo))
(
    spark.read.parquet(str(geo))
    .where((col("order_month") == "2024-03") & (col("country") == "ES"))
    .explain("formatted")
)
```""",
            ),
        md(
            errores(
                [
                    ("Miles de part-000xx", "repartition(200) residual", "repartition(12, order_month) antes del write"),
                    ("Count 2244", "append o join duplicado", "overwrite + dropDuplicates de productos"),
                    ("order_month no está al leer", "API antigua", "spark.read.parquet de 3.5 sí la incluye"),
                ]
            )
        ),
        md(siguiente("../../README.md", "índice del curso")),
    ]
