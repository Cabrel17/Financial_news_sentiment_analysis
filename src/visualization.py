def word_cloud(text, title='wordcloud'):
    """
    This function generate a wordcloud.

    Parameters:
    text : variable of interest
    title (str) : title of the picture
    """
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    full_text = ' '.join(text)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(full_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.show()
