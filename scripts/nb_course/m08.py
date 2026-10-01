"""M08 extra — JSON anidado, schema legacy y facturas embebidas (teoría montada + labs)."""
from __future__ import annotations

from .common import (
    CELDA_0,
    comprueba,
    errores,
    lab_abre,
    md,
    paso,
    prueba,
    reto,
    siguiente,
    teoria_head,
    code,
    boot_cells,
)


def teoria() -> list:
    return [
        md(
            teoria_head(
                "M08 — JSON anidado y schema que cambia (extra)",
                """**Extra.** El pipeline de pedidos (CSV → fact → Parquet) **no** pasa por aquí. Esto cubre lo que M02 no enseña: un JSON de **varios niveles**, bajarlo a columnas, enriquecerlo y **volver a un documento** que una app (o `mongoimport`) pueda comer. Y el otro dolor: el CRM de 2023 y el de 2024 **no tienen las mismas claves**.

En clase ejecutamos **este** fichero, de arriba abajo. El lab es donde construyes tú el mismo flujo sobre los dumps reales.""",
                "../M02-ingesta-preparacion/04-lab-calidad-limpieza.ipynb",
                "02-lab-json-anidado-schema.ipynb",
            )
        ),
        *boot_cells("novashop-clase-m08"),
        md(
            """## Qué hay en `data/raw/` (además de lo de siempre)

NovaShop “tuvo un CRM”:

| Fichero | Qué es | Filas |
|---------|--------|------:|
| `profiles_v1.jsonl` | Dump **2023**, plano (`fullName`, `country`, `email` string) | **100** (C0001–C0100) |
| `profiles_v2.jsonl` | Dump **2024**, anidado (`profile.contact.address.geo`…) | **200** (C0051–C0250) |
| `billing_embedded.jsonl` | Dump de **facturación embebida**: cuenta + array `invoices` (líneas dentro) | **40** (C0001–C0040) |

**50** clientes están en los dos (C0051–C0100): la migración se quedó a medias. v2 trae suciedad: 8 sin `address.country` (el país está en `profile.country`), 6 con `orders_preview` vacío, 5 sin email de trabajo.

`customers.csv` sigue siendo la ficha del pipeline. Estos JSON son **otra** fuente."""
        ),
        md(
            """## Un documento v2 (para no perderse en el schema)

Esto es **un** cliente. Spark, al leer el JSONL, convierte cada llave anidada en `struct` y cada lista en `array`.

```json
{
  "customer_id": "C0051",
  "profile": {
    "contact": {
      "full_name": "Cliente 0051",
      "email": {"work": "c0051@novashop.test", "personal": null},
      "address": {
        "city": "Madrid",
        "country": "ES",
        "geo": {"lat": 40.42, "lon": -3.7}
      }
    },
    "prefs": {"channel": "web", "lang": "es"},
    "country": null
  },
  "orders_preview": [{"id": "X00510", "gmv": 12.5}],
  "meta": {"source": {"system": "crm", "version": 2}}
}
```

Niveles: documento → `profile` → `contact` → `address` → `geo`. Eso es lo que pedían “bastantes niveles”. `products.json` del M02 no llega ni a uno."""
        ),
        md(
            """## Leer v2 e **inferir**: el schema *es* el árbol

JSONL: una línea = un objeto. **No** uses `multiLine` (eso era el array de productos).

Al ejecutar: `count` **200**. `printSchema()` enseña `struct` y `array`. Si ves todo `string` y ningún `struct`, no es este fichero."""
        ),
        code(
            """v2 = spark.read.json(str(RAW / "profiles_v2.jsonl"))
print("v2 filas", v2.count())
v2.printSchema()
v2.select("customer_id", "profile.contact.full_name", "profile.contact.address.geo.lat").show(3, truncate=False)"""
        ),
        md(
            """## Bajar un nivel: el punto (`a.b.c`) no explota filas

`col("profile.contact.address.country")` es **una columna**. Sigue habiendo 200 filas. El árbol no se copia a 200 × N productos.

Al ejecutar: 200 filas; algunos `country` nulos (los 8 sucios). `email.work` nulo en 5."""
        ),
        code(
            """from pyspark.sql.functions import col, coalesce, lit, size, explode, struct, to_json, when, row_number
from pyspark.sql.window import Window

print("filas", v2.count())
print(
    "address.country nulo",
    v2.where(col("profile.contact.address.country").isNull()).count(),
)  # 8
print(
    "email.work nulo",
    v2.where(col("profile.contact.email.work").isNull()).count(),
)  # 5
v2.select(
    "customer_id",
    col("profile.contact.address.country").alias("addr_country"),
    col("profile.country").alias("profile_country"),
).where(col("profile.contact.address.country").isNull()).show()"""
        ),
        md(
            """## `explode`: aquí **sí** cambian las filas

`orders_preview` es un array. `explode` convierte **cada elemento en una fila**. Un cliente con 3 previews pasa a 3 filas. Uno con lista vacía **desaparece** (`explode`); `explode_outer` lo deja con nulos.

Al ejecutar: más de 200 filas (casi todos tienen 1–3 previews; 6 tienen 0 y se caen con `explode`)."""
        ),
        code(
            """prev = v2.select("customer_id", explode("orders_preview").alias("item"))
print("filas tras explode", prev.count())  # > 200
prev.select("customer_id", "item.id", "item.gmv").show(6, truncate=False)
print("clientes que se cayeron (preview vacío)", v2.count() - prev.select("customer_id").distinct().count())  # 6"""
        ),
        md(
            """## Contrato interno: una fila por cliente, columnas planas

Para *transformar* (nombres, país, recuentos) conviene **aplanar**. El país se rescata con el mismo truco que las fechas de M02: `coalesce` de dos sitios + `UNK`.

Al ejecutar: 200 filas; `country` nulo **0** (los 8 sucios salen del `profile.country`). `n_preview` 0 en 6 clientes."""
        ),
        code(
            """v2_flat = v2.select(
    col("customer_id"),
    col("profile.contact.full_name").alias("full_name"),
    coalesce(
        col("profile.contact.address.country"),
        col("profile.country"),
        lit("UNK"),
    ).alias("country"),
    col("profile.contact.email.work").alias("email_work"),
    col("profile.contact.address.city").alias("city"),
    size(col("orders_preview")).alias("n_preview"),
    lit("v2").alias("feed"),
)
print("country nulo", v2_flat.where(col("country").isNull()).count())  # 0
print("n_preview=0", v2_flat.where(col("n_preview") == 0).count())  # 6
v2_flat.show(5, truncate=False)"""
        ),
        md(
            """## Subir otra vez: `struct` + JSON (lo que “come” una app / Mongo)

La aplicación no quiere 8 columnas CSV. Quiere **el árbol**. Montas `struct(...)` y, si hace falta un string, `to_json`. Escribir JSONL es `write.json` (un objeto por fichero-partición; Spark deja un directorio).

Esto **no** es un connector Mongo. Es el documento enriquecido. Un `mongoimport` o un POST a un API usarían ese JSON.

Al ejecutar: una columna `doc` con llaves anidadas; el `show` recorta el string."""
        ),
        code(
            """nested = v2_flat.select(
    "customer_id",
    struct(
        struct(
            col("full_name"),
            struct(col("email_work").alias("work")).alias("email"),
            struct(col("city"), col("country")).alias("address"),
        ).alias("contact"),
        struct(col("n_preview").alias("preview_orders")).alias("stats"),
    ).alias("profile"),
    lit("novashop.profile.v2").alias("schema_id"),
)
nested.printSchema()
nested.select("customer_id", to_json(col("profile")).alias("profile_json")).show(2, truncate=80)

from paths import ensure_dirs

ensure_dirs()
dest = CURATED / "_demo_m08_profiles"
nested.write.mode("overwrite").json(str(dest))
print("escrito", dest)
print("releer", spark.read.json(str(dest)).count())  # 200"""
        ),
        md(
            """## Legacy: v1 no tiene `profile`

Mismo negocio, **otro contrato**. `fullName` vs `full_name`. `email` string vs `email.work`. Si haces `v1.union(v2)` a palo seco, Spark exige las mismas columnas en el mismo orden → peta o rellena basura.

Al ejecutar: v1 **100** filas, schema **plano** (todo al primer nivel)."""
        ),
        code(
            """v1 = spark.read.json(str(RAW / "profiles_v1.jsonl"))
print("v1 filas", v1.count())
v1.printSchema()
v1.show(3, truncate=False)"""
        ),
        md(
            """## Unir las dos épocas: normalizas **cada** feed al mismo contrato

1. Aplanas v1 con los nombres **internos** (`full_name`, `country` vacío → nulo).
2. `unionByName(..., allowMissingColumns=True)` — las columnas que falten se crean nulas.
3. En el solape (50 ids) **gana v2** (`row_number` por `customer_id`, `feed` desc).

Al ejecutar: unión bruta 300 filas; después del “quédate con una ficha por id”: **250**. Eso son todos los clientes de NovaShop. Sin esto, o pierdes a C0001–C0050 (solo v1) o duplicas a C0051–C0100."""
        ),
        code(
            """v1_flat = v1.select(
    col("customer_id"),
    col("fullName").alias("full_name"),
    when(col("country") == "", None).otherwise(col("country")).alias("country"),
    col("email").alias("email_work"),
    lit(None).cast("string").alias("city"),
    lit(0).alias("n_preview"),
    lit("v1").alias("feed"),
)
bruto = v2_flat.unionByName(v1_flat)
print("unión bruta (con duplicados de solape)", bruto.count())  # 300
w = Window.partitionBy("customer_id").orderBy(col("feed").desc())  # v2 antes que v1
profiles = (
    bruto.withColumn("rn", row_number().over(w))
    .where(col("rn") == 1)
    .drop("rn")
)
print("una ficha por cliente", profiles.count())  # 250
print("vienen de v1", profiles.where(col("feed") == "v1").count())  # 50  (C0001–C0050)
print("vienen de v2", profiles.where(col("feed") == "v2").count())  # 200"""
        ),
        md(
            """Eso es “resolver el legacy cuando la migración no se hizo bien”: **un contrato interno**, `coalesce` de sitios distintos, y una regla de precedencia (aquí: el dump nuevo pisa al viejo). Spark no “adivina” el CRM; tú fijas qué gana.

**Lab de esta parte:** [02 JSON anidado y schema](02-lab-json-anidado-schema.ipynb). El pipeline de pedidos no cambia."""
        ),
        md(
            """## Embebido ≠ relacionado (otro dump, misma idea)

Hasta ahora el árbol era *ficha de cliente*. En CMS, Mongo, facturación, CRM “documento”, lo habitual es **meter las facturas dentro de la cuenta**. Un JSON. Cero tabla de facturas. Eso se llama modelo **embebido**.

El modelo **relacionado** es el de M04: `customers.csv` y `orders.csv` son ficheros distintos; el pedido trae `customer_id` y **cruzas**.

| | Relacionado (M04) | Embebido (este dump) |
|--|-------------------|----------------------|
| Cómo viene | 2+ ficheros / tablas | **un** JSON por cuenta, con array `invoices` |
| Dónde está la factura | fila en `orders` | **dentro** de `invoices: [ … ]` |
| Cómo “cruzas” | `join` | primero **partes** el documento; *después* `join` |

Spark no consulta como Mongo (`cuenta.invoices`). Lees el JSONL (un DataFrame con un `array<struct>`), **montas dos DataFrames** (cuentas + facturas) y **luego** cruzas como en M04. El join no está en el JSON; lo armamos **tras la ingesta**.

Fichero: `data/raw/billing_embedded.jsonl` — **40** cuentas (C0001–C0040). **No** es el fact de pedidos. Extra, como el CRM."""
        ),
        md(
            """## Un documento (cuenta + facturas dentro)

```json
{
  "account_id": "C0007",
  "account": {
    "legal_name": "Cliente 0007",
    "vat": "ESB0007",
    "billing": {"city": "Madrid", "country": "ES"}
  },
  "invoices": [
    {
      "invoice_id": "F0007-01",
      "issued": "2024-01-15",
      "status": "paid",
      "currency": "EUR",
      "lines": [{"sku": "P007", "qty": 1, "amount": 10.7}]
    }
  ]
}
```

Tres suciedades a propósito: **5** cuentas con `invoices: []` (C0001–C0005); **3** facturas con `lines: []` (la de C0006 es una); **4** `vat` nulos (C0001–C0004)."""
        ),
        md(
            """## Demo (juguete): un JSON → dos DataFrames → join

Dos cuentas en memoria. Ana tiene 2 facturas; Luis ninguna. Eso cabe en la cabeza.

Al ejecutar:

- `printSchema`: `account` struct, `invoices` array.
- `cuentas` = **2** filas (no explotas).
- `explode` de facturas = **2** filas (Luis **desaparece**).
- `explode_outer` = **3**.
- `left` cuentas ⋈ facturas = **3** (Luis sigue, factura nula). `inner` = **2**.

Eso es “montar dos DataFrames y cruzarlos **después** de ingerir el embebido”."""
        ),
        code(
            """from pyspark.sql.functions import col, size, explode, explode_outer

toy = spark.read.json(spark.sparkContext.parallelize([
    '{"account_id":"A1","account":{"legal_name":"Ana"},"invoices":[{"invoice_id":"F1","total":10.0},{"invoice_id":"F2","total":5.0}]}',
    '{"account_id":"A2","account":{"legal_name":"Luis"},"invoices":[]}',
]))
print("documentos (1 JSON = 1 cuenta)", toy.count())  # 2
toy.printSchema()

# DataFrame 1: grano CUENTA (el array sigue ahí; no lo explotas)
cuentas = toy.select(
    "account_id",
    col("account.legal_name").alias("legal_name"),
    size("invoices").alias("n_inv"),
)
print("cuentas", cuentas.count())  # 2
cuentas.show()

# DataFrame 2: grano FACTURA (explode). Luis se cae.
fact = toy.select("account_id", explode("invoices").alias("inv"))
print("explode facturas", fact.count())  # 2
fact.select("account_id", "inv.invoice_id", "inv.total").show()

print("explode_outer", toy.select("account_id", explode_outer("invoices").alias("inv")).count())  # 3

# Tras la ingesta: cruzar como si hubieran venido relacionados
fact_plana = fact.select(
    "account_id",
    col("inv.invoice_id").alias("invoice_id"),
    col("inv.total").alias("total"),
)
left_ = cuentas.join(fact_plana, "account_id", "left")
inner_ = cuentas.join(fact_plana, "account_id", "inner")
print("left  cuentas ⋈ facturas", left_.count())    # 3
print("inner cuentas ⋈ facturas", inner_.count())   # 2
left_.orderBy("account_id", "invoice_id").show()"""
        ),
        md(
            """**Qué acabas de ver.** El JSON era embebido. Spark no “entra” al array solo. Tú **partes** (cuentas vs facturas) y **cruzas**. El `left` conserva a quien no tiene facturas; el `inner` no.

**Lab de esta parte:** [03 facturas embebidas](03-lab-embedding-facturas.ipynb) — el dump real (`billing_embedded.jsonl`), tres granos (cuenta / factura / línea) y el join tras persistir.

El pipeline `raw` → `fact_lines` **no** usa este fichero."""
        ),
    ]


def lab() -> list:
    return [
        md(
            lab_abre(
                "M08-01",
                "JSON anidado y schema legacy (extra)",
                "M08-01-json-anidado-schema.ipynb",
                """**Extra.** No bloquea M03. Lees los dumps CRM (`profiles_v1` / `v2`), aplanas, unes las dos épocas y escribes un JSON anidado de salida.

Hazlo **después** de M02-02 (ya sabes schema y `coalesce`). Si no has generado datos: `python3 scripts/generate_novashop.py`.""",
                "01-teoria.ipynb",
                "03-lab-embedding-facturas.ipynb",
            )
        ),
        *paso(
                "1",
                "Arranque y los dos dumps",
                "Celda 0 + sesión. Cuento v1 y v2 **antes** de cruzarlos. Si v2 no es 200, el generador es viejo.",
                CELDA_0
                + """

spark = get_spark("novashop-m08")
v1 = spark.read.json(str(RAW / "profiles_v1.jsonl"))
v2 = spark.read.json(str(RAW / "profiles_v2.jsonl"))
print("v1", v1.count(), "v2", v2.count())
v2.printSchema()""",
                "`v1 100` · `v2 200`. Schema de v2 con `struct` (`profile`, `contact`, `address`, `geo`) y `array` (`orders_preview`).",
                "v1 es plano (M02). v2 es el árbol. No uses `multiLine`: son JSONL.",
                if_fail="PATH / count 0: `python3 scripts/generate_novashop.py` y vuelve a la celda.",
            ),
        *paso(
                "2",
                "Bajar el árbol a columnas",
                "`coalesce` del país (address vs profile vs UNK), igual que las dos fechas de M02. `size` del array no explota filas.",
                """from pyspark.sql.functions import (
    col, coalesce, lit, size, explode, struct, to_json, when, row_number,
)
from pyspark.sql.window import Window

v2_flat = v2.select(
    col("customer_id"),
    col("profile.contact.full_name").alias("full_name"),
    coalesce(
        col("profile.contact.address.country"),
        col("profile.country"),
        lit("UNK"),
    ).alias("country"),
    col("profile.contact.email.work").alias("email_work"),
    col("profile.contact.address.city").alias("city"),
    size(col("orders_preview")).alias("n_preview"),
    lit("v2").alias("feed"),
)
print("country nulo", v2_flat.where(col("country").isNull()).count())
print("n_preview=0", v2_flat.where(col("n_preview") == 0).count())
v2_flat.show(3, truncate=False)""",
                "`country` nulo **0**. `n_preview=0` **6**. 200 filas.",
                "Si `country` nulo = 8, no pusiste el `coalesce` de `profile.country`.",
            ),
        *prueba(
                "explode vs size",
                "Cuenta filas con `explode(\"orders_preview\")` y compáralo con `v2.count()`. Luego prueba `explode_outer`.",
                """from pyspark.sql.functions import explode_outer

inner_e = v2.select("customer_id", explode("orders_preview").alias("item"))
outer_e = v2.select("customer_id", explode_outer("orders_preview").alias("item"))
print("explode", inner_e.count(), "distinct clientes", inner_e.select("customer_id").distinct().count())
print("explode_outer", outer_e.count(), "distinct", outer_e.select("customer_id").distinct().count())""",
                "`explode`: distinct clientes **194** (se caen 6). `explode_outer`: distinct **200**. El `size` del paso 2 no cambia el count de clientes.",
            ),
        *paso(
                "3",
                "Aplanar v1 al **mismo** contrato",
                "`fullName` → `full_name`. País `\"\"` → nulo. `city` y `n_preview` no existen en 2023: nulo y 0. Columna `feed=v1`.",
                """v1_flat = v1.select(
    col("customer_id"),
    col("fullName").alias("full_name"),
    when(col("country") == "", None).otherwise(col("country")).alias("country"),
    col("email").alias("email_work"),
    lit(None).cast("string").alias("city"),
    lit(0).alias("n_preview"),
    lit("v1").alias("feed"),
)
v1_flat.printSchema()
print(v1_flat.count())""",
                "100 filas. Mismos nombres de columna que `v2_flat` (si no, el union chilla).",
                "Sin `cast` en `city`, el union puede fallar por tipos.",
            ),
        *paso(
                "4",
                "Unir épocas y quedarte con una ficha",
                "`unionByName` respeta nombres, no el orden. En el solape (50 ids) gana **v2**.",
                """bruto = v2_flat.unionByName(v1_flat)
print("bruto", bruto.count())  # 300
w = Window.partitionBy("customer_id").orderBy(col("feed").desc())
profiles = (
    bruto.withColumn("rn", row_number().over(w))
    .where(col("rn") == 1)
    .drop("rn")
)
print("únicos", profiles.count())  # 250
print("feed v1", profiles.where(col("feed") == "v1").count())  # 50
print("feed v2", profiles.where(col("feed") == "v2").count())  # 200""",
                "**300** → **250**. 50 fichas solo-v1, 200 de v2 (el solape se queda con el árbol 2024).",
                "Si usas `union` clásico y el orden de columnas no coincide, el `email` se mete en `city`.",
            ),
        *prueba(
                "¿Y si gana v1?",
                "Cambia el `orderBy` a `col(\"feed\").asc()` (v1 primero). Cuenta `feed==v2` otra vez. Luego **deja otra vez desc** (v2 gana): esa es la regla de negocio del curso.",
                """w_v1 = Window.partitionBy("customer_id").orderBy(col("feed").asc())
alt = bruto.withColumn("rn", row_number().over(w_v1)).where(col("rn") == 1)
print("si gana v1, filas v2", alt.where(col("feed") == "v2").count())  # 150 (C0101–C0250)
print("regla del curso (gana v2)", profiles.where(col("feed") == "v2").count())  # 200""",
                "Con `asc`: v2 baja a **150**. Markdown: qué clientes perderían el árbol anidado (C0051–C0100).",
            ),
        *paso(
                "5",
                "Escribir el documento enriquecido",
                "Parquet plano en staging (para Spark). JSON anidado en curated (para la app / un import a Mongo). `overwrite` para poder re-ejecutar.",
                """from paths import ensure_dirs

ensure_dirs()
profiles.write.mode("overwrite").parquet(str(STAGING / "profiles_flat"))

nested = profiles.select(
    "customer_id",
    struct(
        struct(
            col("full_name"),
            struct(col("email_work").alias("work")).alias("email"),
            struct(col("city"), col("country")).alias("address"),
        ).alias("contact"),
        struct(col("n_preview").alias("preview_orders")).alias("stats"),
    ).alias("profile"),
    col("feed"),
    lit("novashop.profile.v2").alias("schema_id"),
)
out = CURATED / "profiles_nested"
nested.write.mode("overwrite").json(str(out))
re = spark.read.json(str(out))
print("parquet", spark.read.parquet(str(STAGING / "profiles_flat")).count())
print("json", re.count())
re.printSchema()
re.select("customer_id", "profile.contact.address.country").show(3)""",
                "`parquet` y `json` **250**. El schema releído vuelve a tener `struct`. Una app leería esos JSON; Mongo, un `mongoimport` del directorio.",
                "Spark escribe un **directorio** de JSON, no un único `.json` bonito. Es normal.",
            ),
        md(
            comprueba(
                """v1=100, v2=200, únicos=250. `country` nulo 0 tras coalesce.
JSON relído: 250 y `profile.contact` existe. Markdown: por qué 300 ≠ 250."""
            )
        ),
        *reto(
                "Clientes UNK",
                "Cuenta `country == \"UNK\"` en `profiles`. ¿Vienen de v1, de v2 o de los dos? Markdown con el `feed`.",
                """```python
unk = profiles.where(col("country") == "UNK")
print("UNK", unk.count())
unk.groupBy("feed").count().show()
```""",
            ),
        md(
            errores(
                [
                    ("v2 count ≠ 200", "Generador antiguo", "`python3 scripts/generate_novashop.py`"),
                    ("union AnalysisException", "Nombres/tipos distintos", "Aplana v1 al mismo contrato; `city` con `cast`"),
                    ("250 no sale", "No filtraste rn==1", "Window por customer_id"),
                    ("explode = 200", "Array vacío o no explotaste", "Casi todos tienen 1–3 items; tiene que subir"),
                    ("Un único .json", "Spark escribe carpeta", "Lee con `spark.read.json(dir)`"),
                ]
            )
        ),
        md(siguiente("03-lab-embedding-facturas.ipynb", "M08-02 facturas embebidas")),
    ]


def lab_embed() -> list:
    return [
        md(
            lab_abre(
                "M08-02",
                "Facturas embebidas: partir y cruzar",
                "M08-02-embedding-facturas.ipynb",
                """**Extra.** Un dump de facturación trae la factura **dentro** de la cuenta (modelo embebido, no dos CSV). Tras ingerir: montas **dos DataFrames** (cuentas + facturas) y los **cruzas** como en M04.

No toca `fact_lines`. Hazlo después de M08-01 (ya viste `explode`) o, como mínimo, de M02 + la teoría de este módulo.""",
                "02-lab-json-anidado-schema.ipynb",
                "../M03-transformacion-datos/01-teoria.ipynb",
            )
        ),
        md(
            """## Qué queremos conseguir (léelo antes)

En M04 las facturas **no existen**: hay `orders.csv` y `customers.csv`. El pedido trae `customer_id`. Eso es **relacionado**: dos tablas, un join.

En CMS / Mongo / “el JSON que tira facturación”, lo normal es **un documento por cuenta** y, dentro, `invoices: [ … ]`. La factura **no tiene fichero propio**. Eso es **embebido**.

Spark no hace `cuenta.invoices` como un driver de Mongo. El flujo es:

1. **Ingesta** — lees el JSONL. Un DataFrame. Columna `invoices` = array.
2. **Partir** — un DF a grano cuenta (no explotas) y otro a grano factura (`explode`).
3. **Cruzar** — `join` por `account_id`, *después*. El JSON ya no está; son tablas.

Si solo haces `show()` del documento, “no demuestra nada”: ves un array y no sabes cuántas facturas hay. La prueba son los **counts de grano** (40 cuentas ≠ 75 facturas ≠ 112 líneas).

El encabezado de los pasos dice “con tus palabras”. **Aquí no.** Copia el bloque y rellena `___`.

Fichero: `data/raw/billing_embedded.jsonl`. Si no está: `python3 scripts/generate_novashop.py`."""
        ),
        *paso(
                "1",
                "Ingerir el documento (aún embebido)",
                """Copia y rellena:

```
Paso 1. Un JSON = una cuenta. invoices es un array, no una tabla.
docs = ___ filas (40). printSchema: account struct, invoices array.
vat nulo = ___ (4). Esto todavía NO es un join.
```

JSONL: **no** uses `multiLine` (eso era el array de productos).""",
                CELDA_0
                + """

from pyspark.sql.functions import col, size, explode, explode_outer, sum as fsum
from paths import ensure_dirs

spark = get_spark("novashop-m08")
docs = spark.read.json(str(RAW / "billing_embedded.jsonl"))
print("documentos", docs.count())
docs.printSchema()
print("vat nulo", docs.where(col("account.vat").isNull()).count())
docs.select("account_id", "account.legal_name", size("invoices").alias("n_inv")).show(8)""",
                "`documentos 40`. Schema con `account` struct e `invoices` array de structs (y `lines` dentro). `vat nulo` **4**.",
                "Si el schema es todo string y no hay `array`, no es este fichero (o usaste `multiLine`).",
                if_fail="PATH / 0 filas → `python3 scripts/generate_novashop.py`.",
            ),
        *paso(
                "2",
                "Mirar tres cuentas **sin** explotar (tres casos)",
                """El array se ve distinto según la cuenta. Ejecuta y copia:

```
C0001: invoices = []          → cuenta sin facturas.
C0006: hay factura, lines []  → factura vacía por dentro (otro grano).
C0007: factura con al menos una línea.
```

Esto es el modelo embebido en crudo: **todo en el mismo JSON**.""",
                """print("=== C0001 (sin facturas) ===")
docs.where(col("account_id") == "C0001").select("account_id", "invoices").show(truncate=False)
print("=== C0006 (factura sin líneas) ===")
docs.where(col("account_id") == "C0006").select("account_id", "invoices").show(truncate=False)
print("=== C0007 (factura con línea) ===")
docs.where(col("account_id") == "C0007").select("account_id", "invoices").show(truncate=False)""",
                "C0001 lista vacía. C0006 un `invoice_id` con `lines=[]`. C0007 trae `sku`/`amount`.",
                "Si las tres se ven iguales, estás filtrando mal el `account_id`.",
            ),
        *paso(
                "3",
                "DataFrame 1 — grano **cuenta** (no explotas)",
                """Copia:

```
Paso 3. cuentas tiene ___ filas (40), una por JSON.
n_invoices=0 son ___ (5): C0001–C0005.
size(invoices) NO duplica filas: seguimos a grano cuenta.
```

Esto es “la tabla de clientes” que **montarías** si el origen hubiera sido relacionado.""",
                """cuentas = docs.select(
    col("account_id"),
    col("account.legal_name").alias("legal_name"),
    col("account.vat").alias("vat"),
    col("account.billing.country").alias("country"),
    size("invoices").alias("n_invoices"),
)
print("cuentas", cuentas.count())
print("sin facturas", cuentas.where(col("n_invoices") == 0).count())
cuentas.orderBy("account_id").show(8)""",
                "`cuentas 40`. `sin facturas` **5**. El `show` lista `n_invoices` 0 en las primeras.",
                "`size` cuenta elementos del array; no es `explode`.",
            ),
        *paso(
                "4",
                "DataFrame 2 — grano **factura** (`explode`)",
                """Copia:

```
Paso 4. explode(invoices) = ___ filas (75). Distinct cuentas = ___ (35).
Se cayeron las 5 de invoices [].
Ese DF es la “tabla de facturas” que no venía en un fichero aparte.
```

`account_id` se **repite** en cada factura de la misma cuenta: esa es la clave para el join de después.""",
                """inv_raw = docs.select("account_id", explode("invoices").alias("inv"))
facturas = inv_raw.select(
    "account_id",
    col("inv.invoice_id").alias("invoice_id"),
    col("inv.issued").alias("issued"),
    col("inv.status").alias("status"),
    size("inv.lines").alias("n_lines"),
)
print("facturas", facturas.count())
print("cuentas distintas", facturas.select("account_id").distinct().count())
print("facturas con 0 líneas", facturas.where(col("n_lines") == 0).count())
facturas.where(col("account_id").isin("C0006", "C0007", "C0011")).orderBy("account_id", "invoice_id").show(truncate=False)""",
                "`facturas 75`. Distinct **35**. `n_lines=0` **3** (una es C0006). C0001 **no** sale.",
                "75 ≠ 40: cambió el grano. Si te da 40, no explotaste.",
            ),
        *prueba(
                "explode vs explode_outer (el left del array)",
                """Copia:

```
explode_outer de invoices = ___ (80 = 75 facturas + 5 cuentas vacías).
Las 5 extra tienen invoice_id nulo. Eso es el left “gratis” del array.
```""",
                """outer_inv = docs.select("account_id", explode_outer("invoices").alias("inv"))
print("explode_outer facturas", outer_inv.count())
print(
    "invoice_id nulo",
    outer_inv.where(col("inv.invoice_id").isNull()).count(),
)
outer_inv.where(col("account_id") <= "C0006").select(
    "account_id", col("inv.invoice_id").alias("invoice_id")
).orderBy("account_id").show()""",
                "**80** filas. **5** `invoice_id` nulos (C0001–C0005). C0006 **sí** tiene id (`F0006-01`) aunque `lines` esté vacío.",
            ),
        *paso(
                "5",
                "Todavía más fino: grano **línea** (array dentro del array)",
                """La línea de factura está embebida **otra vez** (`inv.lines`). Mismo truco.

Copia:

```
explode de lines = ___ (112). Distinct facturas = ___ (72 = 75 − 3 vacías).
C0006 desaparece de las líneas (tenía factura, cero lines).
```""",
                """lineas = inv_raw.select(
    "account_id",
    col("inv.invoice_id").alias("invoice_id"),
    explode("inv.lines").alias("line"),
).select(
    "account_id",
    "invoice_id",
    col("line.sku").alias("sku"),
    col("line.qty").alias("qty"),
    col("line.amount").alias("amount"),
)
print("líneas", lineas.count())
print("facturas distintas", lineas.select("invoice_id").distinct().count())
print("¿está C0006?", lineas.where(col("account_id") == "C0006").count())
lineas.where(col("account_id") == "C0007").show()""",
                "`líneas 112`. Facturas distintas **72**. C0006 → **0** filas. C0007 enseña P007 / 10.7.",
                "Tres granos: 40 cuentas, 75 facturas, 112 líneas. Mezclarlos en un `show` del JSON original es lo que “no se entiende”.",
            ),
        *paso(
                "6",
                "Cruzar **después** de ingerir (como si fueran relacionados)",
                """Ya no hay JSON. Hay `cuentas` y `facturas` con `account_id`. Eso es M04.

Copia:

```
inner  = ___ (75): solo quien tiene factura.
left desde cuentas = ___ (80): las 5 sin factura quedan con invoice_id nulo.
left ≠ inner. Si usas inner para un padrón de cuentas, pierdes C0001–C0005.
```""",
                """inner_ = cuentas.join(facturas, "account_id", "inner")
left_ = cuentas.join(facturas, "account_id", "left")
print("inner", inner_.count())
print("left ", left_.count())
print("left con factura nula", left_.where(col("invoice_id").isNull()).count())
left_.where(col("account_id") <= "C0007").orderBy("account_id", "invoice_id").show()""",
                "`inner 75` · `left 80` · nulos de factura **5**. En el `show`, C0001–C0005 salen con `invoice_id` null; C0006 y C0007 no.",
                "El join lo armas tú. El dump embebido **no** traía dos ficheros.",
            ),
        *paso(
                "7",
                "KPI tras el cruce: GMV por cuenta (desde las **líneas**)",
                """El dinero está en `lineas.amount`, no en la cuenta. Agrupas líneas y haces **left** a cuentas: quien no facturó sigue saliendo.

Copia:

```
GMV nulo = ___ (6): 5 sin facturas + C0006 (factura sin líneas).
Cuentas con GMV = 34. El inner del GMV las perdería.
```""",
                """gmv = lineas.groupBy("account_id").agg(fsum("amount").alias("gmv"))
padron = cuentas.join(gmv, "account_id", "left")
print("padrón", padron.count())
print("GMV nulo", padron.where(col("gmv").isNull()).count())
padron.where(col("gmv").isNull()).select("account_id", "n_invoices", "gmv").orderBy("account_id").show()
padron.where(col("account_id") == "C0007").show()""",
                "`padrón 40`. `GMV nulo` **6** (C0001–C0006). C0007 tiene GMV **10.7**.",
                "Si haces inner cuentas ⋈ gmv, C0001–C0006 desaparecen del padrón. Mismo error que el inner de M04 con huérfanos.",
            ),
        *paso(
                "8",
                "Persistir las tablas y volver a cruzar (ya no es JSON)",
                """Escribes tres Parquet en staging. Al releer, el join **no** sabe que un día fueron un documento. Eso es “tratarlo **tras** la ingesta”.

Copia:

```
Releer cuentas/facturas/líneas: 40 / 75 / 112.
left otra vez = 80. El origen embebido ya no está.
```""",
                """ensure_dirs()
cuentas.write.mode("overwrite").parquet(str(STAGING / "billing_accounts"))
facturas.write.mode("overwrite").parquet(str(STAGING / "billing_invoices"))
lineas.write.mode("overwrite").parquet(str(STAGING / "billing_lines"))

c2 = spark.read.parquet(str(STAGING / "billing_accounts"))
f2 = spark.read.parquet(str(STAGING / "billing_invoices"))
l2 = spark.read.parquet(str(STAGING / "billing_lines"))
print("re-cuentas", c2.count(), "re-facturas", f2.count(), "re-líneas", l2.count())
print("left tras Parquet", c2.join(f2, "account_id", "left").count())
c2.join(f2, "account_id", "left").where(col("account_id") == "C0001").show()""",
                "`40 75 112`. left **80**. C0001 releído con `invoice_id` nulo.",
                "A partir de aquí el pipeline es el de siempre (join, KPI, Parquet). El JSON anidado ya hizo su trabajo.",
            ),
        md(
            comprueba(
                """Este bloque, rellenado (no un ensayo):

```
Dump embebido: 1 JSON = 1 cuenta, facturas dentro.
Tras ingesta: cuentas 40, facturas 75, líneas 112.
explode tira las 5 cuentas vacías; explode_outer las deja (80).
inner join 75; left 80. GMV nulo 6 (5 sin factura + C0006 sin líneas).
Parquet 40/75/112; el join posterior ya no ve el JSON.
```"""
            )
        ),
        *reto(
                "Paid vs pending a grano factura",
                "Sobre `facturas` (no sobre líneas): `groupBy(\"status\").count()`. Markdown: ¿cuántas `paid`? No uses `docs` ni `explode` otra vez: ya partiste el documento.",
                """facturas.groupBy("status").count().orderBy("status").show()""",
            ),
        md(
            errores(
                [
                    ("40 facturas", "No explotaste `invoices`", "`explode(\"invoices\")`"),
                    ("explode = 40", "Usaste `size` o no el array", "`size` no duplica filas"),
                    ("C0001 en facturas", "Usaste `explode_outer` y lo llamaste explode", "Inner explode las tira"),
                    ("GMV nulo 5", "Olvidaste C0006 (factura sin líneas)", "El grano línea ≠ grano factura"),
                    ("inner = left", "Todas las cuentas tenían array no vacío", "Este dump tiene 5 `[]`"),
                    ("No está el jsonl", "Generador viejo", "`python3 scripts/generate_novashop.py`"),
                ]
            )
        ),
        md(siguiente("../M03-transformacion-datos/01-teoria.ipynb", "M03 — o vuelve al pipeline de pedidos")),
    ]
