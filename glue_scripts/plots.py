output_dir = "stablerank/sst2/lora_16_sst2"
df = pd.read_csv(os.path.join(output_dir, "sr_layerwise.csv"))
df.head()


# parse kind and stem
def split_kind(name: str):
    if name.endswith(".delta"):     return name[:-6], "delta"
    if name.endswith(".effective"): return name[:-10], "effective"
    if name.endswith(".base"):      return name[:-5], "base"
    return name, "other"

df["stem"], df["kind"] = zip(*df["layer"].map(split_kind))
df = df[df["kind"].isin(["base","delta","effective"])].copy()

# extract module type
def extract_module(stem: str):
    m = re.search(r"attention\.self\.(query|key|value)\.weight$", stem)
    if m: return m.group(1)
    if re.search(r"attention\.output\.dense\.weight$", stem): return "attention.output.dense"
    if re.search(r"intermediate\.dense\.weight$", stem):      return "intermediate.dense"
    if re.search(r"output\.dense\.weight$", stem):            return "output.dense"
    if re.search(r"embeddings\.word_embeddings\.weight$", stem):      return "embeddings.word"
    if re.search(r"embeddings\.position_embeddings\.weight$", stem):  return "embeddings.position"
    if re.search(r"embeddings\.token_type_embeddings\.weight$", stem):return "embeddings.token_type"
    if re.search(r"classifier\.original_module\.dense\.weight$", stem):            return "classifier.dense"
    if re.search(r"classifier\.original_module\.out_proj\.weight$", stem):         return "classifier.out_proj"
    return "other"

df["module"] = df["stem"].map(extract_module)

# order modules (only those present)
preferred_order = [
    "query","key","value",
    "attention.output.dense","intermediate.dense","output.dense",
    "embeddings.word","embeddings.position","embeddings.token_type",
    "classifier.dense","classifier.out_proj","other"
]
present = [m for m in preferred_order if m in set(df["module"])]

# single box plot: 3 boxes per module where available; only base for others
fig = px.box(
    df,
    x="module",
    y="sr",
    color="kind",                                  # base / delta / effective
    category_orders={"module": present, "kind": ["base","delta","effective"]},
    points="all",                                  # set to "all" to show all points
    title="RoBbase (SST2), LoRA Rank 256"
)
fig.update_layout(xaxis_title="", yaxis_title="Stable Rank")

# optional: uncomment if SR ranges are wide
# fig.update_yaxes(type="log")

fig.show()
fig.write_image(f"{output_dir}/sr_boxplot.png", scale=3) 
fig.write_html(f"{output_dir}/sr_boxplot.html")



# parse kind and stem for full fine-tuning
def split_kind_ft(name: str):
   if name.endswith(".pretrained"):  return name[:-11], "pretrained"
   if name.endswith(".finetuned"):   return name[:-10], "finetuned"
   return name, "other"

df["stem"], df["kind"] = zip(*df["layer"].map(split_kind_ft))
df = df[df["kind"].isin(["pretrained", "finetuned"])].copy()

# extract module type
def extract_module(stem: str):
   m = re.search(r"attention\.self\.(query|key|value)\.weight$", stem)
   if m: return m.group(1)
   if re.search(r"attention\.output\.dense\.weight$", stem): return "attention.output.dense"
   if re.search(r"intermediate\.dense\.weight$", stem):      return "intermediate.dense"
   if re.search(r"output\.dense\.weight$", stem):            return "output.dense"
   if re.search(r"embeddings\.word_embeddings\.weight$", stem):      return "embeddings.word"
   if re.search(r"embeddings\.position_embeddings\.weight$", stem):  return "embeddings.position"
   if re.search(r"embeddings\.token_type_embeddings\.weight$", stem):return "embeddings.token_type"
   if re.search(r"classifier\.dense\.weight$", stem):            return "classifier.dense"
   if re.search(r"classifier\.out_proj\.weight$", stem):         return "classifier.out_proj"
   return "other"

df["module"] = df["stem"].map(extract_module)

# order modules (only those present)
preferred_order = [
   "query","key","value",
   "attention.output.dense","intermediate.dense","output.dense",
   "embeddings.word","embeddings.position","embeddings.token_type",
   "classifier.dense","classifier.out_proj","other"
]
present = [m for m in preferred_order if m in set(df["module"])]

# box plot: 2 boxes per module (pretrained vs finetuned)
fig = px.box(
   df,
   x="module",
   y="sr",
   color="kind",                                  # pretrained / finetuned
   category_orders={"module": present, "kind": ["pretrained", "finetuned"]},
   points="all",                                  # show all points
   title="RoBbase (SST2), Full Fine-tuning"
)
fig.update_layout(xaxis_title="", yaxis_title="Stable Rank")

# optional: uncomment if SR ranges are wide
# fig.update_yaxes(type="log")

fig.show()
fig.write_image(f"{output_dir}/sr_boxplot.png", scale=3) 
fig.write_html(f"{output_dir}/sr_boxplot.html")
