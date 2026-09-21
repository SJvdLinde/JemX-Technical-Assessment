# How to label the validation sample

Open `validation_sample.csv` and fill in the **`category`** column for each of the
50 rows. Use one of the eight names below, spelled exactly as written.

These 50 labels are the **ground truth** everything else gets measured against —
the LLM's labels and the trained classifier both get scored against yours. So
label what you actually think the note means, not what you think the code will say.

If a note genuinely could go two ways, use `unclear`. That is a real answer, not
a cop-out, and knowing how often it happens is useful.

---

## The eight categories

### `nothing_reported`
The shift was normal. No reason for extra hours here.

> `ok` · `ntr` · `sharp` · `all quiet` · `as per normal` · `akukho lutho` (isiZulu:
> nothing) · `niks om te rapporteer nie` (Afrikaans: nothing to report)

Blank notes go here too.

---

### `client_requested`
The **client asked for the extra hours** and it was authorised. The client pays
for these, so they are a legitimate business cost, not a problem to fix.

> `requested by centre management for load in, signed off`
> `client asked us to stay for the delivery, ok'd by centre mgmt`
> `CLIENT REQUESTED DEEP CLEAN BEFORE THE AUDIT - APPROVED`
> `klient het ekstra ure gevra vir stocktake` (Afrikaans: client asked for extra hours)

---

### `disputed_client_request`
**The important one.** The note claims client approval **but also states a real
operational cause**. The paperwork says billable; the truth is a failure.

> `client signed for the extra hours but real reason is relief no show agn`

Tells: *but*, *real reason*, *agn/again*. If a note says a client approved it AND
names something that went wrong, this is the category.

---

### `relief_no_show`
**The incoming shift never arrived**, so this person had to stay on. A rostering
failure. Nobody pays for it.

> `Next shift guard did not pitch. Had to cover.`
> `control room says relief coming, nobody came`
> `no relief. stayed. someone must please sort the roster..`
> `aflos het nie opgedaag nie, moes aanbly` (Afrikaans: relief didn't show, had to stay on)
> `next shift akafikanga, ngihlale kuze kube 06h00` (isiZulu: next shift didn't
> arrive, I stayed until 06h00)

**The distinction from `colleague_absent`:** this is about the person who was
supposed to *take over at the end of the shift*.

---

### `colleague_absent`
**A named colleague did not come in**, so this person covered their post or
worked a double. Sick leave, family responsibility, or an unexplained absence.

> `Covering for Ngcobo - booked off sick`
> `Nkosi didnt come in, covered the post`
> `double duty today, Radebe on family responsibility leave`
> `covering sibiya post, no show no call`
> `gedek vir Motaung, siek gemeld` (Afrikaans: covered for Motaung, reported sick)
> `umavuso akezanga namhlaje, ngimele yena` (isiZulu: Mavuso didn't come today,
> I'm standing in for him)

**The distinction from `relief_no_show`:** this is about covering *someone else's
post*, usually with a name attached.

If you find the line between these two genuinely blurry on a given note, that is
worth knowing — use `unclear` and we will count it.

---

### `late_handover`
The shift **ran over because the handover dragged** — paperwork, keys, the OB
book not signed.

> `late handover, waiting on paperwork`
> `waited 25 min for handover, ob book not signed`
> `handover late again, keys missing`
> `oorhandiging was laat, gewag vir sleutels` (Afrikaans: handover was late,
> waited for keys)

---

### `equipment_failure`
**Something broke** and the work took longer as a result.

> `buffer machine kaput, did the floor by hand`
> `scrubber broke down, had to do the floor manually`
> `generator fault, stayed to monitor`
> `lift out of order, everything carried up stairs, took long`
> `gate motor failed, manned it by hand till 6`
> `masjien is stukkend, alles met die hand gedoen` (Afrikaans: machine is broken,
> did everything by hand)

---

### `unclear`
There is text, but it does not explain the hours, or it could genuinely go two
ways and you cannot decide.

> `client says stay till 06h00, dont know if office apprved`

Do not use this as a dumping ground — but do use it when it is honest. If the
categories above do not fit, that is information about the taxonomy, not a
failure on your part.

---

## A note on the spelling

Typos are deliberate and everywhere: `bfufer`, `deelayed`, `siged`, `covr`,
`stocktae`, `Tuesdaay`, `0h600`, `namhlaje`. Read through them. `bfufer machine
kaput` is the same note as `buffer machine kaput`.

Some notes appear more than once with different typos. That is intentional —
label each one on its own terms, and if you give two versions of the same
sentence different labels, that itself tells us something about how hard the
boundary is.

---

## Why this sample looks odd

It is **stratified**, not random: roughly six examples per category, plus ten
deliberately hard ones and four from the long tail. A random 50 would have been
about a third `nothing_reported` and might have contained no equipment failures
at all.

This means the sample is **not representative of the corpus mix** — so overall
accuracy measured on it would be misleading. When reporting, per-category scores
get reweighted back to the true corpus proportions. That is stated in `NOTES.md`.
