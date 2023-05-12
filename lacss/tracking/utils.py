import jax
import jax.numpy as jnp
import numpy as np


def select_sample(chains):
    """a naive sample selection logic
    find the mean cells-per-frame, and return the sample that has the most similar values
    """

    cpf = np.array([np.count_nonzero(s["tracked"], axis=1) for s in chains])
    m_cpf = cpf.mean(axis=1)
    mse = ((cpf - m_cpf[:, None]) ** 2).sum(axis=0)

    idx = np.argmin(mse)
    # plt.plot(cpf[:,idx])

    chain = jax.tree_util.tree_map(lambda v: v[idx], chains)
    # dets = history["detections"]

    return chain


def update_df_with_tracking(df, chains, idx):
    """Create dataframe that record tracking results
    args:
        df: the segmentation dataframe with "index" column
        chains: the smc output
        idx: index of the most likely sample
    returns:
        df: a dataframe
    """

    tracked = chains["tracked"][:, idx, :]
    parent = chains["parent"][:, idx, :]
    age = chains["age"][:, idx, :]

    df_tracking = df.copy()
    df_tracking["parent"] = -1
    df_tracking["tracked"] = False
    df_tracking["age"] = 0

    n_frame = len(tracked)
    ndets = [
        len(df_tracking.loc[df_tracking["frame"] == k + 1]) for k in range(n_frame)
    ]
    first_ids = [
        df_tracking.loc[df_tracking["frame"] == k + 1, "index"].iloc[0]
        for k in range(n_frame)
    ]

    def pad_or_truncate(a, length, constant_values=0):
        if len(a) >= length:
            return a[:length]
        else:
            return np.pad(a, [0, length - len(a)], constant_values=constant_values)

    for k in range(n_frame, 0, -1):
        t = pad_or_truncate(
            tracked[k - 1],
            ndets[k - 1],
        )
        df_tracking.loc[df_tracking["frame"] == k, "tracked"] = t

        p = pad_or_truncate(parent[k - 1], ndets[k - 1], constant_values=-1)
        p = np.where(p >= 0, p + first_ids[k - 2], p)
        df_tracking.loc[df_tracking["frame"] == k, "parent"] = p

        a = pad_or_truncate(
            age[k - 1],
            ndets[k - 1],
        )
        df_tracking.loc[df_tracking["frame"] == k, "age"] = a

    df_tracking = df_tracking.loc[df_tracking["tracked"]].drop("tracked", axis=1)

    df_tracking["cell_id"] = -1

    cell_id = 0
    for id0 in df_tracking.index[::-1]:
        if df_tracking.loc[id0, "cell_id"] != -1:
            continue

        id1 = id0
        while id1 != -1:
            df_tracking.loc[id1, "cell_id"] = cell_id
            if df_tracking.loc[id1, "age"] != 0:
                id1 = df_tracking.loc[id1, "parent"]
            else:
                break
        cell_id += 1

    df_tracking["cell_id"] = df_tracking["cell_id"].max() - df_tracking["cell_id"]

    df_tracking["child_1"] = -1
    df_tracking["child_2"] = -1
    for cell_id in range(df_tracking["cell_id"].max() + 1):
        parent_id = df_tracking.loc[
            df_tracking["cell_id"] == cell_id, "parent"
        ].to_numpy()[0]
        if parent_id != -1:
            parent_cell_id = df_tracking.loc[parent_id, "cell_id"]
            if (
                df_tracking.loc[
                    df_tracking["cell_id"] == parent_cell_id, "child_1"
                ].iloc[0]
                == -1
            ):
                df_tracking.loc[
                    df_tracking["cell_id"] == parent_cell_id, "child_1"
                ] = cell_id
            else:
                df_tracking.loc[
                    df_tracking["cell_id"] == parent_cell_id, "child_2"
                ] = cell_id

    return df_tracking


# def update_with_sample(df, chain, yxs, *, n_frames=-1):

#     df_tracking = df.copy()  # df still has the index column

#     if n_frames <= 0:
#         n_frames = int(df["frame"].max())

#     df_tracking["c0"] = -1
#     df_tracking["c1"] = -1
#     for k in range(n_frames - 1):
#         sel = track[k]["selected"].astype(bool)
#         current_ids = detections[k]["id"][sel]
#         next_ids = detections[k]["id_next"]
#         links, links_div = np.moveaxis(track[k]["next"][sel], -1, 0)
#         df_tracking.loc[current_ids, "c0"] = np.where(links < 0, links, next_ids[links])
#         df_tracking.loc[current_ids, "c1"] = np.where(
#             links_div < 0, links_div, next_ids[links_div]
#         )

#     # label cp
#     df_tracking["cp"] = -1
#     has_child_1 = df_tracking["c0"] > 0
#     child_1_ids = df_tracking.loc[has_child_1]["c0"]
#     df_tracking.loc[child_1_ids, "cp"] = df_tracking.loc[has_child_1][
#         "index"
#     ].to_numpy()
#     has_child_2 = df_tracking["c1"] > 0
#     child_2_ids = df_tracking.loc[has_child_2]["c1"]
#     df_tracking.loc[child_2_ids, "cp"] = df_tracking.loc[has_child_2][
#         "index"
#     ].to_numpy()

#     # remove orphan
#     is_orphan = (
#         (df_tracking["cp"] == -1)
#         & (df_tracking["c0"] == -1)
#         & (df_tracking["c1"] == -1)
#     )
#     df_tracking = df_tracking.loc[~is_orphan]

#     # don't need the index column any more
#     df_tracking = df_tracking.rename({"index": "cell_id"}, axis=1)

#     # recursive tree iteration
#     def id_cells(df, root=0):
#         max_id = 1
#         df["index"] = -1
#         df["parent_idx"] = -1
#         df["child_idx_1"] = -1
#         df["child_idx_2"] = -1

#         def _label(root, cur_id, parent):
#             nonlocal max_id
#             rows = [root]
#             cur_row = root
#             while df.loc[cur_row, "c0"] != -1 and df.loc[cur_row, "c1"] == -1:
#                 cur_row = df.loc[cur_row, "c0"]
#                 rows.append(cur_row)
#             df.loc[rows, "index"] = cur_id
#             df.loc[rows, "parent_idx"] = parent
#             if df.loc[cur_row, "c0"] < 0:  # end of lineage
#                 return
#             else:
#                 max_id += 1
#                 df.loc[rows, "child_idx_1"] = max_id
#                 _label(df.loc[cur_row, "c0"], max_id, cur_id)

#                 new_id_2 = max_id + 1
#                 df.loc[rows, "child_idx_2"] = max_id
#                 _label(df.loc[cur_row, "c1"], max_id, cur_id)

#         _label(root, max_id, -1)

#     id_cells(df_tracking)

#     # df_tracking = df_tracking.drop(['c0', 'c1', 'cp'], axis=1)
#     df_tracking = df_tracking.set_index("index")

#     # the index should now reflect cell_id
#     return df_tracking
