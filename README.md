# scanpy-lowmem

Padrões e patches para rodar Scanpy/Harmony/pySCENIC/infercnvpy em coortes de scRNA-seq
de centenas de milhares de células numa máquina com **16 GB de RAM**, sem cluster.

Extraído de um pipeline de integração de múltiplas coortes públicas de câncer de mama
(~486 mil células antes da subamostragem). Cada técnica aqui resolveu um `MemoryError`
ou incompatibilidade real, documentado no motivo, não só na correção.

## Conteúdo

| Arquivo | Problema que resolve |
|---|---|
| `snippets/qc_mad.py` | QC por MAD (median absolute deviation) em vez de thresholds fixos, quando a qualidade varia muito entre amostras/lotes |
| `snippets/integrate_sparse.py` | `ad.concat` com genes desalinhados (reindex-matmul caro), PCA densificando a matriz HVG, `leidenalg` estourando memória em grafos grandes, e subamostragem estratificada por lote como válvula de escape |
| `snippets/infercnvpy_pandas3_patch.py` | infercnvpy 0.6.1 quebrando com o dtype de string novo do pandas 3.x (fancy-indexing 2D, escrita h5ad, workers de `ProcessPoolExecutor`) |
| `snippets/upsetplot_patches.md` | `upsetplot` 0.9.0 com pandas Copy-on-Write (`fillna(inplace=True)`) e matplotlib mais novo (`np.diff` retornando array em vez de escalar) |

## Por que isso importa

A maioria dos tutoriais de Scanpy assume um servidor com 128GB+ de RAM. Rodar o mesmo
pipeline num notebook de 16GB é um problema de engenharia genuíno, não só de paciência,
e as soluções (evitar densificação, pré-alinhar antes de concatenar, trocar o motor de
clustering, subamostrar de forma estratificada) generalizam para qualquer atlas de
scRNA-seq grande demais para a RAM disponível.

## Uso

Cada snippet é standalone e comentado; não é um pacote instalável. Copie a função que
precisa e adapte os parâmetros (`N_HVG`, `MAX_CELLS_PER_BATCH`, etc.) ao seu dataset e RAM.

## Licença

Licença: MIT. Ver `LICENSE`.
