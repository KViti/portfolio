from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse
from django.urls import reverse
from .models import BugReport, FeatureRequest

# def index(request):
#     return render(request, 'quality_control/index.html')

def bug_list(request):
    bugs = BugReport.objects.all()
    return render(request, 'quality_control/bug_list.html', {'bugs': bugs})


def feature_list(request):
    features = FeatureRequest.objects.all()
    return render(request, 'quality_control/feature_list.html', {'features': features})


from django.views import View
class IndexView(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'quality_control/index.html')

############################################################################################
#############READ#####################READ#######READ################READ###READ##############
############################################################################################
def bug_detail(request, bug_id):
    bug = get_object_or_404(BugReport, id=bug_id)
    return render(request, 'quality_control/bug_detail.html', {'bug': bug})
def feature_detail(request, feature_id):
    feat = get_object_or_404(FeatureRequest, id=feature_id)
    return render(request, 'quality_control/feature_detail.html', {'feat': feat})

from django.views.generic import DetailView
class BugReportDetailView(DetailView):
    model = BugReport
    pk_url_kwarg = "bug_id"
    template_name = 'quality_control/bug_detail.html'
    context_object_name = 'bug'

class FeatureRequestDetailView(DetailView):
    model = FeatureRequest
    pk_url_kwarg = "feature_id"
    template_name = 'quality_control/feature_detail.html'
    context_object_name = 'feat'


############################################################################################
#############CREATE#####################CREATE#######CREATE#################################
############################################################################################
from .forms import BugReportForm, FeatureRequestForm
def add_bug_report(request):
    if request.method == 'POST':
        form = BugReportForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('quality_control:bugs_page')
    else:
        form = BugReportForm()
    return render(request, 'quality_control/bug_create.html', {'form': form})

def add_feat_request(request):
    if request.method == 'POST':
        form = FeatureRequestForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('quality_control:feat_page')
    else:
        form = FeatureRequestForm()
    return render(request, 'quality_control/feature_create.html', {'form': form})

from django.views.generic import CreateView, UpdateView, DeleteView
from django.urls import reverse, reverse_lazy

class BugReportCreateView(CreateView):
    model = BugReport
    form_class = BugReportForm
    template_name = 'quality_control/bug_create.html'
    success_url = reverse_lazy('quality_control:bugs_page')
    context_object_name = 'bug'

class FeatureRequestCreateView(CreateView):
    model = FeatureRequest
    form_class = FeatureRequestForm
    template_name = 'quality_control/feature_create.html'
    success_url = reverse_lazy('quality_control:feat_page')
    context_object_name = 'feature'

############################################################################################
#######UPDATE#############UPDATE###########UPDATE#############UPDATE########UPDATE##########
############################################################################################

def update_bug(request, bug_id):
    bug = get_object_or_404(BugReport, pk=bug_id)
    if request.method == 'POST':
        form = BugReportForm(request.POST, instance=bug)
        if form.is_valid():
            form.save()
            return redirect('quality_control:bug_detail', bug_id=bug_id)
    else:
        form = BugReportForm(instance=bug)
    return render(request, 'quality_control/bug_update.html', {'form': form, 'bug': bug})


def update_feature(request, feature_id):
    feature = get_object_or_404(FeatureRequest, pk=feature_id)
    if request.method == 'POST':
        form = FeatureRequestForm(request.POST, instance=feature)
        if form.is_valid():
            form.save()
            return redirect('quality_control:feature_detail', feature_id=feature_id)
    else:
        form = FeatureRequestForm(instance=feature)
    return render(request, 'quality_control/feature_update.html', {'form': form, 'feature': feature})

class BugReportUpdateView(UpdateView):
    model = BugReport
    form_class = BugReportForm
    template_name = 'quality_control/bug_update.html'
    pk_url_kwarg = 'bug_id'
    success_url = reverse_lazy('quality_control:bug_list')
    context_object_name = 'bug'


class FeatureRequestUpdateView(UpdateView):
    model = FeatureRequest
    form_class = FeatureRequestForm
    template_name = 'quality_control/feature_update.html'
    pk_url_kwarg = 'feature_id'
    success_url = reverse_lazy('quality_control:feature_list')
    context_object_name = 'feat'

############################################################################################
#######DELETE#############DELETE###########DELETE#############DELETE########DELETE##########
############################################################################################

def delete_bug(request, bug_id):
    bug = get_object_or_404(BugReport, pk=bug_id)
    bug.delete()
    return redirect('quality_control:bugs_page')

def delete_feature(request, feature_id):
    feature = get_object_or_404(FeatureRequest, pk=feature_id)
    feature.delete()
    return redirect('quality_control:feat_page')

class BugReportDeleteView(DeleteView):
    model = BugReport
    pk_url_kwarg = 'bug_id'
    success_url = reverse_lazy('quality_control:bugs_page')
    context_object_name = 'bug'
    template_name = 'quality_control/bug_confirm_delete.html'


class FeatureRequestDeleteView(DeleteView):
    model = FeatureRequest
    pk_url_kwarg = 'feature_id'
    success_url = reverse_lazy('quality_control:feat_page')
    context_object_name = 'feature'
    template_name = 'quality_control/feature_confirm_delete.html'