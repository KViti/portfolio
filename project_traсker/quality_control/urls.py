from django.urls import path
from quality_control import views

app_name = 'quality_control'

urlpatterns = [
    # path('', views.index, name='start_quality'),
    path('', views.IndexView.as_view(), name='index'),
    path('bugs/', views.bug_list, name='bugs_page'),
    path('features/', views.feature_list, name='feat_page'),

    path('bugs/new/', views.add_bug_report, name="new_bug"),                                                    #C
    path('features/new/', views.add_feat_request, name="new_feat"),                                             #C
    path('bugs/new/', views.BugReportCreateView.as_view(), name="new_bug"),                                     #C
    path('features/new/', views.FeatureRequestCreateView.as_view(), name="new_feat"),                           #C

    path('bugs/<int:bug_id>/', views.bug_detail, name='bug_detail'),                                            #R
    path('features/<int:feature_id>/', views.feature_detail, name='feature_detail'),                            #R
    path('bugs/<int:bug_id>/', views.BugReportDetailView.as_view(), name='bug_detail'),                         #R
    path('features/<int:feature_id>/', views.FeatureRequestDetailView.as_view(), name='feature_detail'),        #R

    path('bugs/<int:bug_id>/update', views.update_bug, name='update_bug'),                                      #U
    path('features/<int:feature_id>/update', views.update_feature, name='update_feature'),                      #U
    path('bugs/<int:bug_id>/update', views.BugReportUpdateView.as_view(), name='update_bug'),                   #U
    path('features/<int:feature_id>/update', views.FeatureRequestUpdateView.as_view(), name='update_feature'),  #U

    path('bugs/<int:bug_id>/delete', views.delete_bug, name='delete_bug'),                                      #D
    path('features/<int:feature_id>/delete', views.delete_feature, name='delete_feature'),                      #D
    path('bugs/<int:bug_id>/delete', views.BugReportDeleteView.as_view(), name='delete_bug'),                   #D
    path('features/<int:feature_id>/delete', views.FeatureRequestDeleteView.as_view(), name='delete_feature'),  #D
]