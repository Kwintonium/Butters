from django.shortcuts import render
from .core.get_comment_table import main

def home(request):
    return render(request, 'main/index.html')

def results(request):
    search_query = None
    if request.method == 'POST':
        search_query = request.POST.get('youtube_string', None)
        if not search_query: # If no video, use Rick Roll:)
            search_query = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_title, df_html, graph_div, sentiment = main(search_query)
        
    return render(request, 'main/results.html', {'video_title': video_title, 'df_html': df_html, 'graph_div': graph_div, 'sentiment': sentiment})
