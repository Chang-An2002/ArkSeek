/* Optional local management UI. All mutations pass through a concrete preview. */
(() => {
  'use strict';
  let c, csrf='', selectedDoc='';
  const names={profile:'画像管理',profile_restore:'回退画像决定',settings:'保存运行配置',settings_restore:'回退配置',document:'保存资料副本',document_restore:'回退资料版本',bot_start:'启动机器人',bot_stop:'停止机器人',index:'构建/同步L3索引',job_stop:'停止索引任务'};
  const settingLabels={ARKSEEK_LLM_MAX_CONCURRENCY:'LLM并发上限',ARKSEEK_LLM_STALL_TIMEOUT:'流式停滞超时（秒）',ARKSEEK_LLM_TIMEOUT:'慢请求提示（秒）',ARKSEEK_LLM_MAX_DURATION:'请求安全上限（秒）',ARKSEEK_RERANKER:'精排后端'};
  const stages={starting:'启动中',demo_running:'虚构进程运行中',chunking:'切分演示文本',starting_bot:'加载并启动QQ',loading_sources_models:'加载资料与模型，同步索引',done:'已完成',failed:'失败'};
  async function status(){const d=await c.api('manage/status');csrf=d.csrf||'';return d;}
  async function post(action,body){
    const response=await fetch('/api/manage/'+action,{method:'POST',headers:{'Content-Type':'application/json','X-Admin-Token':csrf},body:JSON.stringify(body),cache:'no-store'});
    const d=await response.json();if(!response.ok)throw new Error(d.error||'操作失败');return d;
  }
  function field(label,id,value='',multi=false){const input=c.el(multi?'textarea':'input',{id,value,...(multi?{rows:'7'}:{type:'text'})});if(multi)input.value=value;return {input,node:c.el('div',{class:'field grow'},c.el('label',{for:id},label),input)};}
  function reason(id){return field('操作原因（必填）',id);}
  function blocked(){return c.el('p',{class:'notice'},'当前为只读模式。启用真实管理需在启动时使用 -Live -Manage；演示模式可直接体验虚构管理。');}
  function describe(value){
    if(value===null)return c.el('p',{class:'muted'},'未设置管理员决定 / 无旧版本');
    if(value.stored)return c.el('div',{},c.el('p',{},'原存储：'+value.stored.key+' · '+value.stored.value),c.el('p',{},'管理员决定：'+(value.rule?value.rule.mode==='disabled'?'停用':value.rule.value:'未设置')),c.el('p',{class:'muted'},'原始来源仍保留'));
    if(value.mode)return c.el('p',{},value.mode==='disabled'?'停用该画像':`锁定更正值：${value.value}`);
    if(value.title)return c.el('div',{},c.el('strong',{},value.title),c.el('p',{},value.active?'启用':'停用'),c.el('p',{class:'muted'},`${value.body.length} 字符`),c.el('pre',{class:'source-quote'},value.body.slice(0,400)+(value.body.length>400?'…':'')));
    if(Object.keys(value).some(k=>k in settingLabels))return c.el('div',{},Object.entries(value).map(([k,v])=>c.el('p',{},`${settingLabels[k]||k}：${v}`)));
    if(value.bot)return c.el('p',{},value.bot.alive?'本后台机器人进程正在运行':'本后台机器人进程未运行');
    const label=names[value.operation]||({bot:'启动机器人',index:'构建索引'})[value.operation]||'管理操作';
    return c.el('p',{},label+(value.state==='requested'?' · 已记录操作请求；结果见任务状态':''));
  }
  async function preview(payload){
    try{
      await status();const d=await post('preview',payload);
      const check=c.el('input',{type:'checkbox',id:'confirm-impact'});
      const message=c.el('p',{class:'bad-text',role:'alert'});
      const execute=c.btn('确认执行',async()=>{
        execute.disabled=true;message.textContent='正在执行…';
        try{const result=await post('commit',{token:d.token});c.drawer('操作已记录',c.el('div',{},c.el('p',{},result.message),c.btn('完成并刷新',()=>{document.getElementById('detail-dialog').close();c.load();},'primary')));}
        catch(exc){message.textContent=exc.message+'；请关闭后重新预览。';}
      },'primary');execute.disabled=true;
      check.addEventListener('change',()=>{execute.disabled=!check.checked;});
      c.drawer('操作预览 · '+(names[d.kind]||d.kind),c.el('div',{},c.el('p',{class:'notice'},d.impact),c.el('div',{class:'preview-grid'},c.panel('更改前',describe(d.before)),c.panel('更改后',describe(d.after))),c.el('p',{},'原因：'+d.reason),c.el('p',{class:'muted'},'预览120秒有效；数据或运行状态变化后需要重新预览。'),c.el('label',{class:'confirm-label'},check,'我已核对影响范围'),c.el('div',{class:'toolbar'},execute,c.btn('取消',()=>document.getElementById('detail-dialog').close())),message));
    }catch(exc){c.drawer('未能生成操作预览',c.el('div',{},c.el('p',{class:'bad-text'},exc.message),c.btn('关闭并返回编辑',()=>document.getElementById('detail-dialog').close())));}
  }
  function profileControls(user,fact){
    const box=c.el('div',{class:'management-controls'});
    (async()=>{
      const d=await status();if(!d.enabled){box.append(blocked());return;}
      const value=field('更正后的值','profile-edit-value',fact.value),why=reason('profile-edit-reason');
      const send=mode=>preview({kind:'profile',user_id:user.id,key:fact.key,mode,value:value.input.value,reason:why.input.value});
      box.append(c.el('h3',{},'管理此画像'),c.el('p',{class:'muted'},'只维护当前身份的L1画像。历史原话、L0摘要与L2检索不在本次停用范围；如需彻底遗忘，另行处理。操作前需停止机器人/CLI。'),value.node,why.node,c.el('div',{class:'toolbar'},c.btn('预览更正',()=>send('override')),c.btn('预览停用',()=>send('disabled')),c.btn('解除管理员决定',()=>send('automatic'))));
      const h=await c.api('manage/profile-history',{user_id:user.id,key:fact.key});
      box.append(c.el('h3',{},'管理历史'));
      if(!h.items.length)box.append(c.muted('暂无管理修改记录；这不是此前自动抽取的完整历史。'));
      for(const e of h.items){const after=JSON.parse(e.after_json);box.append(c.el('article',{class:'fact'},c.el('p',{},c.time(e.created_at)+' · '+e.reason),c.muted(after?after.mode==='disabled'?'已停用':'更正为 '+after.value:'已恢复自动抽取'),c.el('p',{},c.btn('预览回退此操作',()=>preview({kind:'profile_restore',user_id:user.id,key:fact.key,event_id:e.id,reason:why.input.value}),'link-button'))));}
    })().catch(exc=>{box.replaceChildren(c.el('p',{class:'bad-text'},exc.message));});
    return box;
  }
  async function eventHistory(events){
    const box=c.panel('操作记录');
    if(!events.length)box.append(c.muted('尚无管理操作；不展示虚构的历史。'));
    for(const e of events)box.append(c.el('div',{class:'fact'},c.el('strong',{},names[e.kind]||e.kind),c.el('p',{},e.reason),c.muted(c.time(e.created_at)),c.el('p',{},c.btn('查看变更',async()=>{
      try{const d=await c.api('manage/event',{id:e.id});const why=reason('event-restore-reason');const nodes=[c.el('p',{},d.reason),c.el('div',{class:'preview-grid'},c.panel('更改前',describe(d.before)),c.panel('更改后',describe(d.after)))];
        if(['settings','document'].includes(e.kind))nodes.push(why.node,c.btn('预览回退此操作',()=>preview({kind:e.kind==='settings'?'settings_restore':'document_restore',event_id:e.id,reason:why.input.value})));
        c.drawer('操作记录',c.el('div',{},...nodes));
      }catch(exc){c.showError(exc);}
    },'link-button'))));
    return box;
  }
  function jobsPanel(d){
    const box=c.panel('任务与进度',c.el('p',{class:'muted'},'显示实际阶段，不虚构百分比。进程存活不等于QQ已就绪；后台重启后不按磁盘PID接管旧进程。'));
    box.append(c.btn('刷新任务状态',async()=>{try{box.replaceWith(jobsPanel(await status()));}catch(exc){c.showError(exc);}}));
    for(const j of d.jobs){const why=reason('job-reason-'+j.id);box.append(c.el('div',{class:'fact'},c.el('strong',{},j.kind==='bot'?'机器人进程':'L3索引任务'),c.el('p',{},({running:'运行中',succeeded:'已结束',failed:'失败',stopped:'已停止',unknown:'状态未确认'})[j.status]||j.status),c.muted(`${c.time(j.created_at)} · ${stages[j.result.stage]||'尚无阶段记录'}`),j.result.chunks!==undefined?c.el('p',{},`${j.result.chunks} 个片段`):null,j.result.note?c.el('p',{class:'muted'},j.result.note):null,j.result.degraded?c.el('p',{class:'bad-text'},'构建存在降级，请在运行日志中核查；不视为完整向量索引验收。'):null,j.result.error_kind?c.el('p',{class:'bad-text'},'异常类型：'+j.result.error_kind):null,j.status==='running'&&j.owned&&j.kind==='index'?c.el('div',{},why.node,c.btn('预览停止任务',()=>preview({kind:'job_stop',id:j.id,reason:why.input.value}))):null));}
    if(!d.jobs.length)box.append(c.muted('尚未通过本后台启动任务。'));
    return box;
  }
  async function settingsPage(){
    const d=await status();if(!d.enabled)return [blocked()];
    const why=reason('runtime-reason');
    const runtime=c.panel('机器人进程',c.el('p',{},d.bot.alive?'本后台所属进程存活；QQ连接与回答能力需另行验证。':'本后台没有运行中的机器人进程；不代表系统中没有终端启动的其他实例。'),d.demo?c.el('p',{class:'notice'},'演示启停只操作虚构进程，不连接QQ、不调用LLM、不读真实配置。'):c.el('p',{class:'notice'},'首次使用管理前，请先关闭终端启动的旧版机器人/CLI。后台不会停止外部实例。'),why.node,c.el('div',{class:'toolbar'},c.btn('预览启动',()=>preview({kind:'bot_start',reason:why.input.value})),c.btn('预览停止',()=>preview({kind:'bot_stop',reason:why.input.value}))));
    const inputs={},form=c.el('form',{onSubmit:e=>{e.preventDefault();preview({kind:'settings',values:Object.fromEntries(Object.entries(inputs).map(([k,n])=>[k,n.value])),reason:configWhy.input.value});}});
    for(const [key,value] of Object.entries(d.settings)){
      if(key==='ARKSEEK_RERANKER'){const select=c.el('select',{id:'config-'+key},['auto','openvino-gpu','openvino-cpu','torch'].map(v=>c.el('option',{value:v},v)));select.value=value;inputs[key]=select;form.append(c.el('div',{class:'field'},c.el('label',{for:'config-'+key},settingLabels[key]),select));}
      else {const f=field(settingLabels[key],'config-'+key,value);f.input.type='number';inputs[key]=f.input;form.append(f.node);}
    }
    const configWhy=reason('config-reason');form.append(configWhy.node,c.el('button',{type:'submit',class:'primary'},'预览保存配置'));
    const config=c.panel('运行配置',c.el('p',{class:'muted'},d.settings_saved?'以下为后台保存的覆盖值，下次启动生效。':'尚未保存后台覆盖值；以下只是模板默认值，不代表当前.env的实际配置。保存后这些项目将覆盖.env，其他项目不变。'),c.el('p',{class:'muted'},'不展示或编辑密钥。安全上限60–900秒且不能低于慢请求提示；不修改思考模式或同群顺序。'),form);
    return [c.el('div',{class:'dashboard-grid'},runtime,config),jobsPanel(d),await eventHistory(d.events.filter(e=>e.kind!=='document'))];
  }
  async function knowledgePage(){
    const d=await status();if(!d.enabled)return [blocked()];
    const list=c.panel('后台资料副本',c.el('p',{class:'muted'},'只管理后台导入的纯文本副本；原始科研文件、Zotero和JC目录均不提供编辑或删除按钮。停用后下次构建/启动退出L3。'),c.btn('新增资料',()=>{selectedDoc='';c.load();}));
    for(const doc of d.documents)list.append(c.btn(c.el('div',{},c.el('strong',{},doc.title),c.el('div',{class:'meta'},`${doc.active?'启用':'停用'} · v${doc.revision} · ${doc.characters} 字符`)),()=>{selectedDoc=doc.id;c.load();},'item'));
    const doc=selectedDoc?await c.api('manage/document',{id:selectedDoc}):{title:'',body:'',active:true};
    const title=field('资料标题','document-title',doc.title),body=field('纯文本内容','document-body',doc.body,true),why=reason('document-reason'),active=c.el('input',{type:'checkbox',id:'document-active'});active.checked=Boolean(doc.active);
    const file=c.el('input',{type:'file',accept:'.txt,.md',id:'document-file'});
    file.addEventListener('change',async()=>{try{const f=file.files[0];if(!f)return;if(f.size>300000)throw new Error('请选择不超过300KB的UTF-8文本文件');body.input.value=await f.text();if(!title.input.value)title.input.value=f.name.replace(/\.(txt|md)$/i,'');}catch(exc){c.showError(exc);}});
    const editor=c.panel(selectedDoc?'编辑项目副本':'新增项目副本',c.el('form',{onSubmit:e=>{e.preventDefault();preview({kind:'document',id:selectedDoc,title:title.input.value,body:body.input.value,active:active.checked,reason:why.input.value});}},title.node,c.el('div',{class:'field'},c.el('label',{for:'document-file'},'可选：读取UTF-8 .txt/.md（仅创建项目副本）'),file),body.node,c.el('label',{class:'confirm-label'},active,'启用此副本'),why.node,c.el('button',{type:'submit',class:'primary'},'预览保存副本')));
    const buildWhy=reason('build-reason');
    const sources=c.panel('既有来源 · 只读状态',c.el('p',{},`本机TXT ${d.sources.local_files.length} 份 · 文献目录缓存${d.sources.catalog_cached?'存在':'未建立'} · 外部来源配置${d.sources.external_configured?'存在':'未单独设置'}`),...d.sources.local_files.map(f=>c.el('p',{},`${f.name} · ${f.bytes} 字节`)),c.el('p',{class:'muted'},'这里只检查项目文件元数据与缓存是否存在，未连接Zotero、未核实全文可用性；现有原始文件不提供修改按钮。'));
    return [c.el('p',{class:'notice'},d.demo?'当前只维护虚构资料；演示构建验证文本切分，不加载真实模型。':'资料维护需要机器人/CLI停止。构建会读取现有L3配置与资料并加载模型，操作前会显示影响预览。'),sources,c.el('div',{class:'split'},list,editor),c.panel('L3索引维护',c.el('p',{class:'muted'},'沿用现有分段、BM25、向量和精排流程。更新派生索引，不删除原始资料；机器人下次启动加载新版本。'),buildWhy.node,c.btn('预览构建/同步索引',()=>preview({kind:'index',reason:buildWhy.input.value}))),jobsPanel(d),await eventHistory(d.events.filter(e=>e.kind==='document'))];
  }
  window.ArkManage={init:context=>{c=context;},profileControls,settingsPage,knowledgePage,status};
})();
