/* Small responsive charts: no network resources; all drill-downs are read-only. */
(() => {
  'use strict';
  const NS='http://www.w3.org/2000/svg';
  const node=(tag,attrs={},text=null)=>{const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==null)n.textContent=text;return n;};
  const localHour=b=>b.hour===null?b.label:String(b.hour).padStart(2,'0')+'时';
  function trend(series,latency,onSelect) {
    const root=document.createElement('div');root.className='trend-chart';
    const svg=node('svg',{role:'img','aria-label':latency?'每个时段总等待与排队中位数，点击时段查看请求':'分时请求量，点击时段查看请求'});
    const note=document.createElement('p');note.className='chart-readout';note.setAttribute('aria-live','polite');
    note.textContent=latency?'空缺表示该时段没有可用耗时；两条线均为各自样本的中位数。':'点击有记录的时段查看请求；0表示该时段没有监测记录。';
    root.append(svg,note);
    function draw() {
      const width=Math.max(240,root.clientWidth||360),height=208,left=48,right=18,top=18,bottom=35;
      svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.replaceChildren();
      const fields=latency?['median_ms','queue_median_ms']:['count'];
      const values=series.flatMap(b=>fields.map(f=>b[f])).filter(v=>Number.isFinite(v)&&v>=0);
      if(!values.length || !series.some(b=>b.count>0)) {svg.append(node('text',{x:width/2,y:height/2,'text-anchor':'middle',class:'chart-axis'},'暂无可绘制的监测记录'));return;}
      const max=Math.max(...values,latency?1000:1)*1.1;
      const inner=width-left-right;
      const x=i=>left+inner*(i+.5)/series.length;
      const y=v=>height-bottom-(height-top-bottom)*v/max;
      for(let t=0;t<=2;t++) {
        const v=max*t/2,py=y(v);
        svg.append(node('line',{x1:left,x2:width-right,y1:py,y2:py,class:'chart-grid'}),node('text',{x:left-8,y:py+4,'text-anchor':'end',class:'chart-axis'},latency?(v/1000).toFixed(v>=10000?0:1):String(Math.round(v))));
      }
      svg.append(node('text',{x:7,y:13,class:'chart-axis'},latency?'秒':'条'));
      const labelCount=width<360?3:5;
      const indexes=new Set(Array.from({length:Math.min(series.length,labelCount)},(_,i)=>Math.round(i*(series.length-1)/Math.max(1,Math.min(series.length,labelCount)-1))));
      for(const i of indexes)svg.append(node('text',{x:x(i),y:height-9,'text-anchor':'middle',class:'chart-axis'},localHour(series[i])));
      if(latency) {
        for(const [fi,field] of fields.entries()) {
          let path='',open=false;
          series.forEach((b,i)=>{const v=b[field];if(v===null || !Number.isFinite(v)){open=false;return;}path+=(open?' L ':' M ')+x(i)+' '+y(v);open=true;});
          svg.append(node('path',{d:path,class:'chart-line series-'+fi}));
          series.forEach((b,i)=>{if(Number.isFinite(b[field]))svg.append(node('circle',{cx:x(i),cy:y(b[field]),r:3.5,class:'chart-point series-'+fi}));});
        }
      }else {
        series.forEach((b,i)=>{if(b.count>0)svg.append(node('rect',{x:x(i)-Math.min(14,inner/series.length*.32),y:y(b.count),width:Math.min(28,inner/series.length*.64),height:Math.max(2,y(0)-y(b.count)),rx:3,class:'chart-column'}));});
      }
      series.forEach((b,i)=>{
        if(!b.count)return;
        const description=b.day+(b.hour===null?'':` ${b.hour}:00`)+`，${b.count}条请求`+(latency?`，总等待中位数${b.median_ms===null?'未记录':(b.median_ms/1000).toFixed(1)+'秒'}，排队中位数${b.queue_median_ms===null?'未记录':(b.queue_median_ms/1000).toFixed(1)+'秒'}`:'');
        const hit=node('rect',{x:x(i)-inner/series.length/2,y:top,width:inner/series.length,height:height-top-bottom,fill:'transparent',role:'button',tabindex:0,'aria-label':description+'；查看请求',class:'chart-hit'});
        hit.append(node('title',{},description));
        hit.addEventListener('mouseenter',()=>{note.textContent=description;});
        hit.addEventListener('focus',()=>{note.textContent=description;});
        hit.addEventListener('click',()=>onSelect(b));
        hit.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onSelect(b);}});
        svg.append(hit);
      });
    }
    // The observer is attached after insertion; detached views release it.
    requestAnimationFrame(()=>{if(!root.isConnected)return;draw();const observer=new ResizeObserver(()=>{if(!root.isConnected){observer.disconnect();return;}draw();});observer.observe(root);});
    return root;
  }
  window.ArkCharts={trend};
})();
