# Plotly-based PCA and t-SNE visualizations with fixed colors by ASC-sorted labels.
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative as qual
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PLOTLY_PALETTE = (
    qual.Plotly + qual.D3 + qual.Set1 + qual.Set2 + qual.Set3 + qual.Dark24 + qual.Light24
)

def _sorted_unique_labels(labels):
    return sorted(np.unique(labels).tolist())

def _label2color(labels):
    uniq = _sorted_unique_labels(labels)
    return {lab: PLOTLY_PALETTE[i % len(PLOTLY_PALETTE)] for i, lab in enumerate(uniq)}

def pca_viz_embs(embeds, labels, n_componets=None, title='', legend_title='', html_output_file='',
                 filter_labels=None, label2color=None, pca=None):
    
    embeds = np.asarray(embeds)
    labels = np.asarray(labels)
    if pca is None:
        pca = PCA(n_components=n_componets, random_state=42)
        emb = pca.fit_transform(embeds)
    else:
        emb = pca.transform(embeds)

    uniq_labels = _sorted_unique_labels(labels)

    if label2color is None:  # setting colors before filtering labels
        l2c = _label2color(uniq_labels)
    else:
        l2c = label2color

    if filter_labels is not None:
        uniq_labels = [lab for lab in uniq_labels if lab in filter_labels]
    
    
    fig = go.Figure()

    for chap in uniq_labels:
        mask = labels == chap
        if n_componets == 2:
            fig.add_trace(go.Scatter(x=emb[mask,0], y=emb[mask,1], mode='markers',
                                     name=str(chap), marker=dict(size=4, color=l2c[chap]),
                                     hovertext=[str(chap)]*int(mask.sum()), hoverinfo='text'))
        elif n_componets == 3:
            fig.add_trace(go.Scatter3d(x=emb[mask,0], y=emb[mask,1], z=emb[mask,2], mode='markers',
                                       name=str(chap), marker=dict(size=3, color=l2c[chap]),
                                       hovertext=[str(chap)]*int(mask.sum()), hoverinfo='text'))
        else:
            return
    if n_componets == 2:
        layout_kwargs = dict(xaxis_title='PC1', yaxis_title='PC2')
    elif n_componets == 3:
        layout_kwargs = dict(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
    else:
        return
    fig.update_layout(title_text=title, legend_title_text=legend_title, width=1200, height=900, **layout_kwargs)
    if html_output_file:
        fig.write_html(html_output_file)
        print(f'📄 Saved plot to {html_output_file}')
    return fig, l2c, pca

def tsne_viz_embs(embeds, labels, n_componets=None, title='', legend_title='', html_output_file='',
                  random_state=42, perplexity=50, max_iter=1000, learning_rate='auto',
                  filter_labels=None, label2color=None, pca=None, tsne=None):
    
    embeds = np.asarray(embeds)
    labels = np.asarray(labels)

    if pca is None:
        pca = PCA(n_components=n_componets, random_state=42)
        embeds = pca.fit_transform(embeds)
    else:
        embeds = pca.transform(embeds)

    if tsne is None:    
        tsne = TSNE(n_components=n_componets, random_state=random_state, perplexity=perplexity, n_jobs=-3,
                max_iter=max_iter, learning_rate=learning_rate, init='pca', metric='euclidean')
        emb = tsne.fit_transform(embeds)
    else:
        emb = tsne.transform(embeds)

    fig = go.Figure()

    uniq_labels = _sorted_unique_labels(labels)

    if label2color is None: # setting colors before filtering labels
       l2c = _label2color(uniq_labels)
    else:
        l2c = label2color

    if filter_labels is not None:
        uniq_labels = [lab for lab in uniq_labels if lab in filter_labels]
    

    for chap in uniq_labels:
        mask = labels == chap
        if n_componets == 2:
            fig.add_trace(go.Scatter(x=emb[mask,0], y=emb[mask,1], mode='markers',
                                     name=str(chap), marker=dict(size=4, color=l2c[chap]),
                                     hovertext=[str(chap)]*int(mask.sum()), hoverinfo='text'))
        elif n_componets == 3:
            fig.add_trace(go.Scatter3d(x=emb[mask,0], y=emb[mask,1], z=emb[mask,2], mode='markers',
                                       name=str(chap), marker=dict(size=3, color=l2c[chap]),
                                       hovertext=[str(chap)]*int(mask.sum()), hoverinfo='text'))
        else:
            return
    if n_componets == 2:
        layout_kwargs = dict(xaxis_title='t-SNE 1', yaxis_title='t-SNE 2')
    elif n_componets == 3:
        layout_kwargs = dict(scene=dict(xaxis_title='t-SNE 1', yaxis_title='t-SNE 2', zaxis_title='t-SNE 3'))
    else:
        return
    fig.update_layout(title_text=title, legend_title_text=legend_title, width=1200, height=900, **layout_kwargs)
    if html_output_file:
        fig.write_html(html_output_file)
        print(f'📄 Saved plot to {html_output_file}')
    return fig, l2c, tsne
