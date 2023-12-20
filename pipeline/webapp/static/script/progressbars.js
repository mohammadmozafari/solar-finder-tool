const downloadController = document.querySelector('.downloadProgressInput');
const processController = document.querySelector('.processProgressInput');
const downloadRadialProgress = document.querySelector('.DownloadRadialProgress');
const processRadialProgress = document.querySelector('.ProcessRadialProgress');

const setDownloadProgress = (progress) => {
  const value = `${progress}%`;
  downloadRadialProgress.style.setProperty('--progress', value)
  downloadRadialProgress.innerHTML = value
  downloadRadialProgress.setAttribute('aria-valuenow', value)
}

const setProcessProgress = (progress) => {
  const value = `${progress}%`;
  processRadialProgress.style.setProperty('--progress', value)
  processRadialProgress.innerHTML = value
  processRadialProgress.setAttribute('aria-valuenow', value)
}

setDownloadProgress(downloadController.value)
setProcessProgress(processController.value)
