function listpngfiles() { 
    var FilesDirectory = '/uploads/png/';

// get auto-generated page 
$.ajax({url: FilesDirectory}).then(function(html) {
    // create temporary DOM element
    var document = $(html);

    // find all links ending with .png 
    document.find('a[href$=.png]').each(function() {
        var pngName = $(this).text();
        var pngUrl = $(this).attr('href');

        // do what you want here 
    })
});
}