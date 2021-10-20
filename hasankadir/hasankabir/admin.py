from django.contrib import admin

from hasankabir.models import Database, Results

# Register your models here.

class YourModelAdmin(admin.ModelAdmin):
    pass

admin.site.register(Results, YourModelAdmin)
admin.site.register(Database, YourModelAdmin)